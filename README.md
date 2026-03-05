# NEXA — Site Officiel

Site vitrine de l'application NEXA, plateforme IA de gestion pastorale pour les éleveurs africains.

## Structure

```
nexa-site/
├── index.html          ← Site complet (single-page)
├── app.py              ← Serveur FastAPI
├── requirements.txt    
├── screenshots/        ← Mettre ici les captures d'écran de l'app
│   ├── screen1.png
│   ├── screen2.png
│   └── ...
└── .python-version     ← Python 3.11
```

## Ajouter les vraies screenshots

Dans `index.html`, remplace les `.screen-placeholder` par :
```html
<img src="/screenshots/screen1.png" alt="Écran 1"/>
```

## Déploiement Render

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Python: 3.11

## Liens

- Démo web : https://nexa-tcwn.onrender.com
- GitHub : https://github.com/SabriMD/nexa-grazing
