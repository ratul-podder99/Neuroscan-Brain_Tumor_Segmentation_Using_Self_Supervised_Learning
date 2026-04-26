# NeuroScan — Brain Tumour Segmentation Web App

A Flask web application that serves your BraTS2020 U-Net model with a clinical dark-theme UI.

## 📁 Project Structure
```
brain_tumor_app/
├── app.py                 ← Flask backend + inference logic
├── best_unet_model.h5     ← ⬅ PUT YOUR MODEL FILE HERE
├── requirements.txt
├── templates/
│   └── index.html         ← Frontend UI
└── README.md
```

## 🚀 Quick Start

### 1 · Place your model file
Copy `best_unet_model.h5` (from Kaggle `/kaggle/working/`) into this folder:
```
brain_tumor_app/best_unet_model.h5
```

### 2 · Install dependencies
```bash
cd brain_tumor_app
pip install -r requirements.txt
```

### 3 · Run the server
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧠 Input Modes

| Mode | What to upload | How it works |
|---|---|---|
| **Single Image** | Any PNG/JPG MRI slice | R→T1, G→T1ce, B→T2, mean→FLAIR |
| **4 Modalities** | 4 separate PNG/JPG files | Each mapped directly to T1/T1ce/T2/FLAIR |

For best accuracy, use the **4 Modalities** mode with the actual four MRI channels exported as PNG slices.

## 📊 Output

- **Original Input** — the T1ce channel of your upload
- **Segmentation Mask** — colour-coded class prediction
- **Overlay** — mask blended onto the scan
- **Class Stats** — coverage (%) and confidence for each class
- **Heatmaps** — per-class probability maps (inferno scale)

### Class Colours
| Class | Colour | Description |
|---|---|---|
| 0 | Dark | Background |
| 1 | Red `#e63946` | Necrotic Core |
| 2 | Amber `#f4a261` | Peritumoral Edema |
| 3 | Teal `#06d6a0` | Enhancing Tumour |

## ⚙️ Demo Mode
If no `.h5` file is found, the app launches in **Demo Mode** and shows a synthetic segmentation so you can preview the UI without a model.

## 📝 Notes
- This is a research tool, not a clinical diagnostic device.
- The model was trained on BraTS 2020 (128×128 2D slices, 4 modalities).
- Input images are automatically resized to 128×128.
