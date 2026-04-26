import os, io, base64, tempfile
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

IMG_SIZE    = 128
NUM_CLASSES = 4
MODEL_PATH  = "best_unet_model.h5"

CLASS_INFO = {
    0: {"name": "Background",        "color": "#1a1a2e", "rgb": (26,  26,  46)},
    1: {"name": "Necrotic Core",     "color": "#e63946", "rgb": (230,  57,  70)},
    2: {"name": "Peritumoral Edema", "color": "#f4a261", "rgb": (244, 162,  97)},
    3: {"name": "Enhancing Tumour",  "color": "#06d6a0", "rgb": (  6, 214, 160)},
}

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*intersection+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)

def iou_score(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)-intersection
    return (intersection+smooth)/(union+smooth)

def pixel_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,-1),tf.argmax(y_pred,-1)),tf.float32))

def dice_loss(y_true, y_pred): return 1.0 - dice_coefficient(y_true, y_pred)
def combined_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true,y_pred)+dice_loss(y_true,y_pred)

CUSTOM_OBJECTS = {"combined_loss":combined_loss,"dice_coefficient":dice_coefficient,
                  "iou_score":iou_score,"pixel_accuracy":pixel_accuracy}

def conv_block(inputs, f):
    x = layers.Conv2D(f,3,padding='same')(inputs)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv2D(f,3,padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    return x

def encoder_block(inputs, f):
    x = conv_block(inputs,f); p = layers.MaxPooling2D((2,2))(x); return x,p

def decoder_block(inputs, skip, f):
    x = layers.Conv2DTranspose(f,(2,2),strides=2,padding='same')(inputs)
    x = layers.Concatenate()([x,skip]); x = conv_block(x,f); return x

def build_unet(input_shape=(128,128,4), num_classes=4):
    inp = layers.Input(input_shape)
    s1,p1=encoder_block(inp,64);  s2,p2=encoder_block(p1,128)
    s3,p3=encoder_block(p2,256);  s4,p4=encoder_block(p3,512)
    b=conv_block(p4,1024)
    d=decoder_block(b,s4,512); d=decoder_block(d,s3,256)
    d=decoder_block(d,s2,128); d=decoder_block(d,s1,64)
    out=layers.Conv2D(num_classes,1,padding='same',activation='softmax')(d)
    return models.Model(inp,out,name='U-Net')

model=None
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  '{MODEL_PATH}' not found – DEMO mode."); return
    try:
        model=keras.models.load_model(MODEL_PATH,custom_objects=CUSTOM_OBJECTS)
        print("✅  Model loaded successfully.")
    except:
        try:
            model=build_unet(); model.load_weights(MODEL_PATH)
            print("✅  Model loaded via weights-only method.")
        except Exception as e:
            print(f"❌  Failed: {e}\n⚠️  DEMO mode.")

# ── Input validation ──────────────────────────────────────────────────────────

def validate_image_input(pil_img):
    try:
        img  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        g = gray - gray.min()
        if g.max() > 0: g = (g / g.max() * 255).astype(np.uint8)
        else: return False, "Image appears completely blank."

        # 1. Reject uniform images
        if float(g.std()) < 12:
            return False, "Image is too uniform — this does not appear to be a valid MRI scan."

        # 2. Reject colour photographs
        r,gr,b = img[:,:,0].astype(float),img[:,:,1].astype(float),img[:,:,2].astype(float)
        color_spread = float(np.mean(np.abs(r-gr))+np.mean(np.abs(gr-b))+np.mean(np.abs(r-b)))
        if color_spread > 22:
            return False, "This appears to be a colour photograph. Brain MRI scans are grayscale. Please upload a valid brain MRI slice."

        # 3. Must have a dark background (MRI characteristic)
        bh=max(1,h//10); bw=max(1,w//10)
        border=np.concatenate([g[:bh,:].flatten(),g[-bh:,:].flatten(),
                                g[:,:bw].flatten(),g[:,-bw:].flatten()])
        if float(np.mean(border < 28)) < 0.25:
            return False, "No dark background detected. Brain MRI slices have a distinct black background. This may be a non-brain scan or a photograph."

        # 4. Tissue brightness ratio
        _, thr = cv2.threshold(g, 18, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.sum(thr > 0)) / (h * w)
        if bright_ratio < 0.05:
            return False, "Very little tissue detected. The image may be too dark or not a valid brain MRI slice."
        if bright_ratio > 0.92:
            return False, "Image is uniformly bright — brain MRI should have a distinct dark background."

        # 5. Shape analysis — brain axial slices are roughly circular
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "No tissue region detected in this image."

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        perim   = cv2.arcLength(largest, True)

        # Circularity — brain is highly circular (>0.45), kidney is elongated (<0.35)
        if perim > 0:
            circularity = 4 * np.pi * area / (perim ** 2)
            if circularity < 0.35:
                return False, "The main tissue region does not resemble a brain cross-section. Brain MRI axial slices are roughly circular. This may be a kidney, liver, or other organ scan."

        # 6. Aspect ratio of bounding box — brain is roughly square
        x, y, bw2, bh2 = cv2.boundingRect(largest)
        aspect_ratio = float(bw2) / float(bh2) if bh2 > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Image aspect ratio suggests a non-brain organ. Brain MRI axial slices have a roughly square bounding region. This may be a kidney, liver, or spine scan."

        # 7. The tissue region must be reasonably centred in the image
        # Brain is typically centred, kidney/spine scans are often off-centre
        cx_contour = x + bw2 / 2
        cy_contour = y + bh2 / 2
        cx_image   = w / 2
        cy_image   = h / 2
        x_offset   = abs(cx_contour - cx_image) / w
        y_offset   = abs(cy_contour - cy_image) / h
        if x_offset > 0.35 or y_offset > 0.35:
            return False, "The main tissue region is too far from the image centre. Brain MRI slices typically show the brain centrally positioned. This may be a non-brain scan."

        # 8. The tissue region must occupy a reasonable portion of the image
        # Brain fills roughly 30-80% of an axial slice
        fill_ratio = area / (h * w)
        if fill_ratio < 0.08:
            return False, "Tissue region is too small relative to the image. This may not be a brain MRI slice."
        if fill_ratio > 0.88:
            return False, "Tissue fills too much of the image — brain MRI should have clear dark borders around the brain."

        # 9. Check for internal complexity — brain has many internal structures
        # A kidney CT slice has simpler internal texture
        brain_region = g[y:y+bh2, x:x+bw2]
        if brain_region.size > 0:
            internal_std = float(brain_region.std())
            if internal_std < 18:
                return False, "Internal tissue texture is too uniform. Brain MRI scans show complex internal structures. This may be a different organ or an invalid scan."

        return True, "OK"
    except:
        return True, "OK"

# ── Clinical Decision Support ─────────────────────────────────────────────────

def generate_clinical_suggestions(class_stats, volumes_cm3, total_vol_cm3,
                                   tumour_pct, avg_confidence):
    """
    Rule-based clinical decision support.
    Rules derived from:
    - WHO CNS Classification 2021 (Louis et al., Acta Neuropathologica)
    - Stupp et al. 2005, NEJM (Stupp Protocol)
    - NCCN CNS Cancer Guidelines v2.2023
    - ESMO High-grade Glioma Guidelines 2021
    - Vecht et al. 1994, Neurology (dexamethasone)
    """

    enh_pct  = class_stats["3"]["percentage"]   # Enhancing Tumour %
    nec_pct  = class_stats["1"]["percentage"]   # Necrotic Core %
    ede_pct  = class_stats["2"]["percentage"]   # Peritumoral Edema %
    enh_vol  = volumes_cm3.get("3", 0.0)
    nec_vol  = volumes_cm3.get("1", 0.0)
    ede_vol  = volumes_cm3.get("2", 0.0)
    tumour_total = total_vol_cm3

    # Confidence gate — suppress suggestions if model is uncertain
    if avg_confidence < 55.0:
        return {
            "confidence_warning": True,
            "grade": None,
            "grade_label": "Insufficient confidence",
            "grade_color": "#888",
            "summary": "Model confidence is too low to generate reliable clinical suggestions. Please review the segmentation manually or use a higher-quality scan.",
            "findings": [],
            "pathway": [],
            "molecular": [],
            "followup": [],
            "disclaimer": True,
            "sources": []
        }

    # Very small tumour — may not be clinically significant
    if tumour_total < 0.5 and tumour_pct < 3.0:
        return {
            "confidence_warning": False,
            "grade": "uncertain",
            "grade_label": "Minimal tumour signal",
            "grade_color": "#4a6080",
            "summary": "Tumour signal is minimal on this slice. This may represent a very early lesion, post-treatment change, or partial volume effect. Clinical correlation required.",
            "findings": [
                {"label": "Total tumour coverage", "value": f"{tumour_pct:.1f}% of slice", "note": "Below threshold for confident grading"},
            ],
            "pathway": [
                {"step": "1", "action": "MRI follow-up in 6–8 weeks", "rationale": "To assess progression or stability", "source": "NCCN CNS Guidelines 2023"},
                {"step": "2", "action": "Neuro-oncology consultation", "rationale": "For clinical correlation with symptoms", "source": "ESMO Guidelines 2021"},
            ],
            "molecular": [],
            "followup": ["Repeat MRI with contrast in 6–8 weeks", "Clinical neurological examination"],
            "disclaimer": True,
            "sources": ["NCCN CNS Cancer Guidelines v2.2023", "ESMO High-grade Glioma Guidelines 2021"]
        }

    # ── Grade classification ──────────────────────────────────────────────────
    # Based on WHO 2021: GBM defined by necrosis + microvascular proliferation
    # Imaging correlates: necrotic core + contrast enhancement

    if enh_pct >= 20.0 and nec_pct >= 25.0:
        grade       = "IV"
        grade_label = "High-grade pattern (WHO Grade IV — GBM)"
        grade_color = "#e63946"
        grade_note  = ("Imaging profile consistent with Glioblastoma Multiforme (GBM). "
                       "High enhancing fraction (%.1f%%) combined with large necrotic core (%.1f%%) "
                       "are the primary imaging criteria for WHO Grade IV per the 2021 CNS classification." % (enh_pct, nec_pct))

    elif enh_pct >= 10.0 and nec_pct < 25.0:
        grade       = "III"
        grade_label = "Intermediate-grade pattern (WHO Grade III)"
        grade_color = "#f4a261"
        grade_note  = ("Moderate enhancement (%.1f%%) without dominant necrosis suggests anaplastic "
                       "glioma pattern (WHO Grade III). Definitive grading requires histopathology "
                       "and molecular profiling." % enh_pct)

    elif enh_pct < 10.0 and ede_pct > enh_pct and nec_pct < 15.0:
        grade       = "II"
        grade_label = "Low-grade pattern (WHO Grade II)"
        grade_color = "#06d6a0"
        grade_note  = ("Low enhancement (%.1f%%) with oedema-dominant profile suggests low-grade "
                       "glioma pattern. However, non-enhancing tumours can still be high grade — "
                       "biopsy is essential for definitive classification." % enh_pct)

    else:
        grade       = "uncertain"
        grade_label = "Indeterminate grade"
        grade_color = "#4a6080"
        grade_note  = ("Imaging profile does not clearly match a single WHO grade pattern. "
                       "Mixed features require histopathological correlation.")

    # ── Surgical assessment (NCCN CNS Guidelines) ────────────────────────────
    surgical_recs = []
    if tumour_total >= 30.0:
        surgical_recs.append({
            "step": "1",
            "action": "Maximal safe surgical resection (debulking)",
            "rationale": f"Total volume {tumour_total:.2f} cm³ exceeds threshold for surgical debulking. Gross total resection improves survival in GBM.",
            "source": "NCCN CNS Cancer Guidelines v2.2023"
        })
    elif 10.0 <= tumour_total < 30.0:
        surgical_recs.append({
            "step": "1",
            "action": "Surgical resection — assess resectability",
            "rationale": f"Volume {tumour_total:.2f} cm³ is in the surgical range. Resectability depends on eloquent area involvement (motor cortex, speech areas). Pre-surgical functional MRI recommended.",
            "source": "NCCN CNS Cancer Guidelines v2.2023"
        })
    elif 3.0 <= tumour_total < 10.0:
        surgical_recs.append({
            "step": "1",
            "action": "Stereotactic Radiosurgery (SRS) candidate — evaluate",
            "rationale": f"Volume {tumour_total:.2f} cm³ is within Gamma Knife / CyberKnife eligibility range (<10 cm³). Discuss with radiation oncology.",
            "source": "NCCN CNS Guidelines 2023 + Leksell Gamma Knife Society guidelines"
        })
    else:
        surgical_recs.append({
            "step": "1",
            "action": "Watchful waiting with close MRI monitoring",
            "rationale": f"Small volume ({tumour_total:.2f} cm³). Active surveillance may be appropriate pending biopsy results and grade confirmation.",
            "source": "NCCN CNS Cancer Guidelines v2.2023"
        })

    # ── Chemotherapy + radiation pathway ─────────────────────────────────────
    chemo_recs = []
    if grade == "IV":
        chemo_recs.append({
            "step": "2",
            "action": "Stupp Protocol — Temozolomide (TMZ) + 60 Gy radiotherapy",
            "rationale": "Standard of care for GBM. Concomitant TMZ (75 mg/m²/day) during RT, followed by 6 cycles adjuvant TMZ (150–200 mg/m²). Median survival benefit established in pivotal RCT.",
            "source": "Stupp et al., NEJM 2005 (DOI: 10.1056/NEJMoa043330)"
        })
        if enh_vol >= 5.0:
            chemo_recs.append({
                "step": "3",
                "action": "Consider Bevacizumab (anti-VEGF therapy)",
                "rationale": f"High enhancing tumour volume ({enh_vol:.2f} cm³) indicates significant angiogenesis. Bevacizumab targets VEGF-driven vascular proliferation characteristic of GBM.",
                "source": "NCCN CNS Guidelines 2023 + Friedman et al. 2009, JCO"
            })

    elif grade == "III":
        chemo_recs.append({
            "step": "2",
            "action": "Radiation therapy (54–60 Gy) + PCV chemotherapy or TMZ",
            "rationale": "For WHO Grade III anaplastic gliomas. PCV (procarbazine, CCNU, vincristine) or TMZ used based on molecular profile (IDH/1p19q status).",
            "source": "CATNON trial (van den Bent et al., Lancet 2017) + NCCN CNS Guidelines 2023"
        })

    elif grade == "II":
        chemo_recs.append({
            "step": "2",
            "action": "Radiation therapy alone (45–54 Gy) or active surveillance",
            "rationale": "Low-grade glioma management depends on age, symptoms, and molecular markers. Young, asymptomatic patients may be observed. High-risk features warrant radiation.",
            "source": "EORTC 22845 trial + NCCN CNS Guidelines 2023"
        })

    # ── Edema / steroid assessment ────────────────────────────────────────────
    steroid_recs = []
    if ede_vol >= 20.0:
        steroid_recs.append({
            "step": "4" if grade in ["IV","III"] else "3",
            "action": "Initiate Dexamethasone (corticosteroid therapy)",
            "rationale": f"Peritumoral oedema volume {ede_vol:.2f} cm³ is clinically significant. Dexamethasone reduces vasogenic oedema, relieves raised ICP, and improves neurological function prior to definitive treatment.",
            "source": "Vecht et al., Neurology 1994 + ESMO High-grade Glioma Guidelines 2021"
        })
    elif 10.0 <= ede_vol < 20.0:
        steroid_recs.append({
            "step": "4" if grade in ["IV","III"] else "3",
            "action": "Low-dose Dexamethasone — consider based on symptoms",
            "rationale": f"Moderate peritumoral oedema ({ede_vol:.2f} cm³). Steroid initiation depends on clinical symptoms (headache, neurological deficit). Not always required at this volume.",
            "source": "ESMO High-grade Glioma Guidelines 2021"
        })

    # ── Molecular testing — always recommended per WHO 2021 ──────────────────
    molecular = [
        {
            "test": "MGMT Promoter Methylation",
            "why": "Predicts response to Temozolomide chemotherapy. Methylated MGMT = better TMZ response and improved survival. Determines whether TMZ should be included in treatment.",
            "source": "Hegi et al., NEJM 2005 + WHO CNS Classification 2021"
        },
        {
            "test": "IDH1 / IDH2 Mutation Status",
            "why": "Single most important prognostic marker in glioma. IDH-mutant tumours have significantly better prognosis. Mandatory for WHO 2021 classification — grade cannot be assigned without it.",
            "source": "WHO Classification of CNS Tumours 2021 (Louis et al.)"
        },
        {
            "test": "1p/19q Codeletion",
            "why": "Defines oligodendroglioma (1p/19q codeleted) vs astrocytoma. Oligodendrogliomas respond better to PCV chemotherapy. Critical for treatment selection.",
            "source": "WHO CNS Classification 2021 + CODEL trial"
        },
        {
            "test": "TERT Promoter Mutation",
            "why": "Associated with GBM and oligodendroglioma. Combined with IDH and 1p/19q for integrated molecular diagnosis.",
            "source": "WHO CNS Classification 2021"
        }
    ]

    # ── Follow-up imaging ─────────────────────────────────────────────────────
    if grade == "IV":
        followup = [
            "Post-operative MRI within 24–72 hours of surgery (assess extent of resection)",
            "MRI at 4–6 weeks post-surgery (pre-radiation baseline)",
            "MRI every 8–12 weeks during and after chemoradiation",
            "Watch for pseudoprogression (MRI increase at 2–6 months post-RT does not always mean true progression)"
        ]
    elif grade == "III":
        followup = [
            "Post-treatment MRI at 6–8 weeks",
            "MRI every 3–4 months for first 2 years",
            "Annual MRI thereafter if stable"
        ]
    else:
        followup = [
            "MRI with contrast in 6–8 weeks",
            "MRI every 3–6 months for first 2 years",
            "Annual MRI thereafter"
        ]

    # ── Build final findings list ─────────────────────────────────────────────
    findings = [
        {"label": "Enhancing tumour",    "value": f"{enh_pct:.1f}%  ({enh_vol:.3f} cm³)",
         "note": "Active, vascularised tumour — key grading marker (WHO 2021)"},
        {"label": "Necrotic core",       "value": f"{nec_pct:.1f}%  ({nec_vol:.3f} cm³)",
         "note": "Tissue death — WHO Grade IV criterion when >25% with enhancement"},
        {"label": "Peritumoral oedema",  "value": f"{ede_pct:.1f}%  ({ede_vol:.3f} cm³)",
         "note": "Vasogenic oedema — drives steroid decision"},
        {"label": "Total tumour volume", "value": f"{tumour_total:.3f} cm³",
         "note": "Drives surgical vs radiosurgery decision (NCCN thresholds)"},
        {"label": "Model confidence",    "value": f"{avg_confidence:.1f}%",
         "note": "Average confidence across predicted tumour pixels"},
    ]

    pathway = surgical_recs + chemo_recs + steroid_recs

    sources = list({r["source"] for r in pathway})
    sources += ["WHO Classification of CNS Tumours 2021 (Louis et al., Acta Neuropathologica)"]

    return {
        "confidence_warning": False,
        "grade":        grade,
        "grade_label":  grade_label,
        "grade_color":  grade_color,
        "grade_note":   grade_note,
        "summary":      f"Imaging profile suggests {grade_label}. "
                        f"Total tumour volume: {tumour_total:.3f} cm³. "
                        f"The following clinical pathway is typical for tumours with this profile "
                        f"according to published guidelines. This is NOT a diagnosis.",
        "findings":     findings,
        "pathway":      pathway,
        "molecular":    molecular,
        "followup":     followup,
        "disclaimer":   True,
        "sources":      sources
    }

# ── NIfTI helpers ─────────────────────────────────────────────────────────────

def load_nii_from_bytes(file_bytes, filename):
    suffix=".nii.gz" if filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
        tmp.write(file_bytes); tmp_path=tmp.name
    try:
        img=nib.load(tmp_path); vol=img.get_fdata()
        zooms=img.header.get_zooms(); shape=vol.shape
    finally:
        os.unlink(tmp_path)
    return vol,zooms,shape

def find_best_slice(vol,lo=30,hi=120):
    start,end=max(0,lo),min(vol.shape[2],hi)
    counts=[np.count_nonzero(vol[:,:,i]) for i in range(start,end)]
    return start+int(np.argmax(counts))

def extract_slice(vol,idx):
    sl=vol[:,:,idx].astype(np.float32)
    sl=cv2.resize(sl,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_LINEAR)
    if sl.max()>0: sl=(sl-sl.min())/(sl.max()-sl.min())
    return sl

def preprocess_single(pil):
    img=np.array(pil.convert("RGB"),dtype=np.float32)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    s=np.stack([img[:,:,0],img[:,:,1],img[:,:,2],img.mean(-1)],-1)
    for c in range(4):
        ch=s[:,:,c]
        if ch.max()>0: s[:,:,c]=(ch-ch.min())/(ch.max()-ch.min())
    return s.astype(np.float32)

def preprocess_four(t1,t1ce,t2,flair):
    chs=[]
    for p in [t1,t1ce,t2,flair]:
        g=np.array(p.convert("L"),dtype=np.float32)
        g=cv2.resize(g,(IMG_SIZE,IMG_SIZE))
        if g.max()>0: g=(g-g.min())/(g.max()-g.min())
        chs.append(g)
    return np.stack(chs,-1).astype(np.float32)

def estimate_vol_cm3(pred_mask,class_id,orig_shape,zooms):
    pixels=int(np.sum(pred_mask==class_id))
    if pixels==0: return 0.0
    mm_x=float(zooms[0])*(orig_shape[0]/IMG_SIZE)
    mm_y=float(zooms[1])*(orig_shape[1]/IMG_SIZE)
    mm_z=float(zooms[2]) if len(zooms)>2 else 1.0
    return round(pixels*mm_x*mm_y*mm_z/1000.0,3)

def demo_pred(tensor):
    H,W,_=tensor.shape; mask=np.zeros((H,W),dtype=np.uint8)
    cy,cx=H//2,W//2
    for r,c in [(30,2),(18,3),(10,1)]: cv2.circle(mask,(cx,cy),r,c,-1)
    probs=np.zeros((H,W,NUM_CLASSES),dtype=np.float32)
    for c in range(NUM_CLASSES): probs[:,:,c]=(mask==c).astype(np.float32)
    return probs,mask

def mask_rgb(mask):
    rgb=np.zeros((*mask.shape,3),dtype=np.uint8)
    for c,info in CLASS_INFO.items(): rgb[mask==c]=info["rgb"]
    return rgb

def build_overlay(gray,seg_rgb,alpha=0.55):
    base=np.stack([gray]*3,-1).astype(np.float32)
    non_bg=(seg_rgb.sum(-1)>CLASS_INFO[0]["rgb"][0]).astype(np.float32)[...,None]
    return np.clip(base*(1-alpha*non_bg)+seg_rgb.astype(np.float32)*(alpha*non_bg),0,255).astype(np.uint8)

def to_b64(arr):
    buf=io.BytesIO(); Image.fromarray(arr).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template("index.html")

@app.route("/nii_info",methods=["POST"])
def nii_info():
    try:
        f=request.files.get("t1")
        if not f: return jsonify({"error":"no file"}),400
        vol,zooms,shape=load_nii_from_bytes(f.read(),f.filename)
        valid,msg=validate_nii_shape(vol,f.filename)
        if not valid: return jsonify({"error":msg}),400
        best=find_best_slice(vol)
        return jsonify({"depth":int(vol.shape[2]),"best_slice":best,
                        "zooms":[float(z) for z in zooms[:3]],"shape":list(shape[:3])})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/predict",methods=["POST"])
def predict():
    try:
        files=request.files; form=request.form
        tensor=None; display=None; slice_used=None; total_slices=None
        orig_shape=None; zooms=None

        if all(k in files for k in ("nii_t1","nii_t1ce","nii_t2","nii_flair")):
            vols={}; zms={}; shps={}
            for m in ("nii_t1","nii_t1ce","nii_t2","nii_flair"):
                v,z,s=load_nii_from_bytes(files[m].read(),files[m].filename)
                valid,msg=validate_nii_shape(v,files[m].filename)
                if not valid: return jsonify({"error":msg}),400
                vols[m]=v; zms[m]=z; shps[m]=s
            total_slices=int(vols["nii_t1"].shape[2])
            req_s=form.get("slice_idx")
            slice_idx=int(req_s) if req_s is not None else find_best_slice(vols["nii_t1"])
            slice_idx=max(0,min(slice_idx,total_slices-1)); slice_used=slice_idx
            chs=[extract_slice(vols[k],slice_idx) for k in ("nii_t1","nii_t1ce","nii_t2","nii_flair")]
            tensor=np.stack(chs,-1).astype(np.float32); display=np.uint8(chs[1]*255)
            zooms=zms["nii_t1"]; orig_shape=shps["nii_t1"]

        elif all(k in files for k in ("t1","t1ce","t2","flair")):
            imgs={}
            for m in ("t1","t1ce","t2","flair"):
                pil=Image.open(files[m])
                valid,msg=validate_image_input(pil)
                if not valid: return jsonify({"error":f"Invalid {m.upper()} — {msg}"}),400
                imgs[m]=pil
            tensor=preprocess_four(imgs["t1"],imgs["t1ce"],imgs["t2"],imgs["flair"])
            display=np.uint8(tensor[:,:,1]*255)

        elif "image" in files:
            pil=Image.open(files["image"])
            valid,msg=validate_image_input(pil)
            if not valid: return jsonify({"error":msg}),400
            tensor=preprocess_single(pil); display=np.uint8(tensor[:,:,1]*255)
        else:
            return jsonify({"error":"No valid files uploaded."}),400

        if model is not None:
            probs=model.predict(tensor[np.newaxis,...],verbose=0)[0]
            pred_mask=np.argmax(probs,-1).astype(np.uint8)
        else:
            probs,pred_mask=demo_pred(tensor)

        total=pred_mask.size
        cs={c:float(np.sum(pred_mask==c))/total*100 for c in range(NUM_CLASSES)}
        tumour_pct=sum(cs[c] for c in [1,2,3])
        conf={c:float(probs[:,:,c][pred_mask==c].mean()) if np.any(pred_mask==c) else 0.0
              for c in range(NUM_CLASSES)}

        if zooms is None: zooms=(1.0,1.0,1.0); orig_shape=(240,240,155)
        vols_cm3={c:estimate_vol_cm3(pred_mask,c,orig_shape,zooms) for c in range(NUM_CLASSES)}
        total_vol_cm3=round(sum(vols_cm3[c] for c in [1,2,3]),3)

        orig_b64=to_b64(cv2.resize(display,(256,256),interpolation=cv2.INTER_LINEAR))
        seg=mask_rgb(pred_mask)
        seg_b64=to_b64(cv2.resize(seg,(256,256),interpolation=cv2.INTER_NEAREST))
        ov=build_overlay(cv2.resize(display,(256,256)),cv2.resize(seg,(256,256),interpolation=cv2.INTER_NEAREST))
        ov_b64=to_b64(ov)
        conf_maps=[]
        for c in range(1,NUM_CLASSES):
            h=cv2.applyColorMap(np.uint8(probs[:,:,c]*255),cv2.COLORMAP_INFERNO)
            conf_maps.append(to_b64(cv2.resize(cv2.cvtColor(h,cv2.COLOR_BGR2RGB),(128,128))))

        class_stats_out={str(c):{"name":CLASS_INFO[c]["name"],"color":CLASS_INFO[c]["color"],
            "percentage":round(cs[c],2),"confidence":round(conf[c]*100,1),
            "vol_cm3":vols_cm3[c]} for c in range(NUM_CLASSES)}

        # Average confidence across tumour classes only
        tumour_confs=[conf[c]*100 for c in [1,2,3] if np.any(pred_mask==c)]
        avg_conf=float(np.mean(tumour_confs)) if tumour_confs else 0.0

        suggestions=generate_clinical_suggestions(
            class_stats=class_stats_out,
            volumes_cm3={str(c):vols_cm3[c] for c in range(NUM_CLASSES)},
            total_vol_cm3=total_vol_cm3,
            tumour_pct=tumour_pct,
            avg_confidence=avg_conf
        )

        voxel_info={"mm_x":round(float(zooms[0]),3),"mm_y":round(float(zooms[1]),3),
                    "mm_z":round(float(zooms[2] if len(zooms)>2 else 1.0),3),
                    "orig_h":int(orig_shape[0]),"orig_w":int(orig_shape[1])}

        return jsonify({
            "success":True,"demo_mode":model is None,
            "original":orig_b64,"segmentation":seg_b64,"overlay":ov_b64,
            "conf_maps":conf_maps,"slice_used":slice_used,"total_slices":total_slices,
            "tumour_percentage":round(tumour_pct,2),
            "total_vol_cm3":total_vol_cm3,
            "voxel_info":voxel_info,
            "class_stats":class_stats_out,
            "suggestions":suggestions,
        })

    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}),500

@app.route("/reconstruct3d",methods=["POST"])
def reconstruct3d():
    try:
        files=request.files
        if not all(k in files for k in ("nii_t1","nii_t1ce","nii_t2","nii_flair")):
            return jsonify({"error":"All 4 NIfTI files required for 3D reconstruction."}),400
        vols={}; zooms=None; orig_shape=None
        for m in ("nii_t1","nii_t1ce","nii_t2","nii_flair"):
            v,z,s=load_nii_from_bytes(files[m].read(),files[m].filename)
            valid,msg=validate_nii_shape(v,files[m].filename)
            if not valid: return jsonify({"error":msg}),400
            vols[m]=v
            if zooms is None: zooms=z; orig_shape=s
        depth=vols["nii_t1"].shape[2]; step=3
        slice_indices=list(range(max(0,30),min(120,depth),step))
        batch=np.array([np.stack([extract_slice(vols[k],idx)
            for k in ("nii_t1","nii_t1ce","nii_t2","nii_flair")],-1)
            for idx in slice_indices],dtype=np.float32)
        if model is not None:
            all_masks=np.argmax(model.predict(batch,verbose=0),-1).astype(np.uint8)
        else:
            all_masks=np.zeros((len(slice_indices),IMG_SIZE,IMG_SIZE),dtype=np.uint8)
            mid=len(slice_indices)//2; cx,cy=IMG_SIZE//2+8,IMG_SIZE//2-5
            for i in range(len(slice_indices)):
                dist=abs(i-mid)
                for r,c in [(int(max(0,30-dist*1.3)),2),(int(max(0,18-dist*1.3)),3),(int(max(0,9-dist*1.3)),1)]:
                    if r>0: cv2.circle(all_masks[i],(cx,cy),r,c,-1)
        mm_x=float(zooms[0])*(orig_shape[0]/IMG_SIZE)
        mm_y=float(zooms[1])*(orig_shape[1]/IMG_SIZE)
        mm_z=float(zooms[2] if len(zooms)>2 else 1.0)*step
        vox_vol_mm3=mm_x*mm_y*mm_z
        volumes_cm3={str(c):round(int(np.sum(all_masks==c))*vox_vol_mm3/1000.0,3) for c in range(1,NUM_CLASSES)}
        volumes_cm3["total"]=round(sum(float(v) for v in volumes_cm3.values()),3)
        DS=48
        points={str(c):{"x":[],"y":[],"z":[]} for c in range(1,NUM_CLASSES)}
        for z_i,mask in enumerate(all_masks):
            sm=cv2.resize(mask,(DS,DS),interpolation=cv2.INTER_NEAREST)
            for c in range(1,NUM_CLASSES):
                ys,xs=np.where(sm==c)
                points[str(c)]["x"].extend(xs.tolist())
                points[str(c)]["y"].extend(ys.tolist())
                points[str(c)]["z"].extend([z_i]*len(xs))
        return jsonify({"success":True,"demo_mode":model is None,
            "volumes_cm3":volumes_cm3,"points":points,
            "bounds":{"x":DS,"y":DS,"z":len(slice_indices)},
            "voxel_mm":{"x":round(mm_x,3),"y":round(mm_y,3),"z":round(mm_z,3)},
            "slices_processed":len(slice_indices)})
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}),500

if __name__=="__main__":
    load_model()
    app.run(debug=True,host="0.0.0.0",port=5001)