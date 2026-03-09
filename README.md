# 🌿 NEXA — AI-Powered Pasture Intelligence

> *One photo. One analysis. One plan.*  
> Transforming ancestral livestock knowledge into measurable financial assets through AI.

[![Live Demo](https://img.shields.io/badge/🌍_Official_Site-Live-22C55E?style=for-the-badge)](https://nexa-tcwn.onrender.com)
[![Model](https://img.shields.io/badge/Model-EfficientNet--B2_ONNX-blue?style=for-the-badge)](./nexa_biomasse.onnx)
[![Accuracy](https://img.shields.io/badge/R²_Accuracy-86%25-22C55E?style=for-the-badge)]()
[![Hackathon](https://img.shields.io/badge/RabHacks-Franco--African_2026-gold?style=for-the-badge)]()

---

## 🎯 The Problem

50 million livestock farmers across the Sahel manage millions of animals with **no tools**:

- **10,000+ deaths/year** from pasture resource conflicts (Mali, Niger, Burkina, Chad)
- **234% collateral** required for loans in Ethiopia — farmers excluded from credit
- **9.5M animals lost** during the Horn of Africa drought — from lack of anticipation

Current methods are either too slow, too unreliable, or completely out of reach for small farmers.

---

## 💡 The Solution

NEXA turns a **smartphone photo** into a complete grazing plan — offline, in seconds.

```
📸 Photo → 🧠 AI Analysis → 📋 Rotation Plan
```

Two modes for two realities:
- 🏡 **Sedentary farmers** — rotation plans, soil health score, carbon credit tracking
- 🐪 **Nomadic farmers** — real-time NDVI satellite zones showing best vegetation nearby

---

## 📁 Repository Structure

```
NEXA_site/
│
├── index.html              # Official landing page (deployed on Render)
├── app.py                  # FastAPI demo backend
├── requirements.txt        # Python dependencies
│
├── nexa_biomasse.onnx      # Trained EfficientNet-B2 model (ONNX format)
├── convertir_modele.py     # PyTorch → ONNX conversion script
│
├── assets/
│   ├── logo-nexa.png
│   ├── background.png
│   └── partenaire*.png
│
└── screenshots/            # App screenshots for carousel
    ├── screen1.png
    ├── screen2.png
    └── screen3.png
```

---

## 🧠 AI Model

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B2 |
| Task | Pasture biomass regression (g/m²) |
| Format | ONNX (cross-platform, offline-ready) |
| R² Accuracy | **86%** |
| Input | RGB image (224×224) |
| Output | Biomass estimate (GDM/ha) |
| Dataset | Australian pastures — multi-season, multi-region, multi-species |

### Run inference locally

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("nexa_biomasse.onnx")

# Preprocess image
img = Image.open("pasture.jpg").resize((224, 224))
x = np.array(img).astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 224, 224)

# Inference
biomass = session.run(None, {"input": x})[0]
print(f"Estimated biomass: {biomass[0][0]:.1f} g/m²")
```

### Convert model (PyTorch → ONNX)

```bash
python convertir_modele.py
```

---

## 🌐 Web Demo

The FastAPI backend serves the ONNX model via a REST API.

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### API Endpoint

```
POST /predict
Content-Type: multipart/form-data
Body: image (file)

Response:
{
  "biomasse_gdm": 1240.5,
  "jours_disponibles": 12,
  "statut": "bon",
  "score_sante": 72.3
}
```

---

## 🚀 Deploy on Render

This repo is configured for **Static Site** deployment on Render:

```
Build Command   : (empty)
Publish Directory: .
```

The `index.html` includes the full landing page with:
- Hero section + animated stats
- How it works (4 steps)
- App showcase carousel
- Features grid
- **Growth Dashboard** (Semaine 3 traction data)
- Revenue model
- Partners section

---

## 📱 Mobile App

The Flutter mobile app lives in a separate repository:  
👉 [github.com/SabriMD/nexa-grazing](https://github.com/SabriMD/nexa-grazing)

**Stack:** Flutter · Firebase Auth · Hive · ONNX Runtime · OpenWeatherMap API · flutter_map

---

## 📊 Traction — Week 3

| Metric | Value |
|--------|-------|
| LinkedIn impressions | ~200 |
| Instagram followers | 12 (48h) |
| Direct field contacts | 8 |
| Positive feedback | 4 |
| Markets identified | 7 countries |

---

## 🤝 Team

Built in **3 weeks** at the **Franco-African Hackathon RabHacks 2026**

---

## 📄 License

MIT License — see [LICENSE](./LICENSE)

---

<div align="center">
  <strong>🌍 50 million farmers deserve better tools.</strong><br/>
  <a href="https://nexa-tcwn.onrender.com">nexa-tcwn.onrender.com</a>
</div>
