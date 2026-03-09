"""
NEXA — Serveur de démonstration
Estimation de biomasse des pâturages à partir de photos
"""

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# ============================================================
# INITIALISATION
# ============================================================
app = FastAPI(title="NEXA — Estimation Biomasse Pâturages")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Chargement du modèle ONNX au démarrage
print("📂 Chargement du modèle ONNX...")
session = ort.InferenceSession("modeles/nexa_biomasse.onnx")
NOM_ENTREE = session.get_inputs()[0].name
TAILLE_IMAGE = (260, 260)  # Taille EfficientNet-B2
print("✅ Modèle ONNX chargé — serveur prêt")

# ============================================================
# PRÉ-TRAITEMENT DE L'IMAGE
# ============================================================
def pretraiter_image(image_pil: Image.Image) -> np.ndarray:
    """
    Transforme une image PIL en tenseur numpy normalisé.
    Même pipeline que l'entraînement PyTorch.
    """
    # Redimensionner
    image = image_pil.convert('RGB').resize(TAILLE_IMAGE)
    
    # Convertir en array numpy [H, W, C] → [C, H, W]
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    # Normalisation ImageNet standard
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Ajouter dimension batch [1, C, H, W]
    return img_array[np.newaxis, :].astype(np.float32)


# ============================================================
# CALCUL DU PLAN DE ROTATION
# ============================================================
def calculer_plan_rotation(gdm_g_par_m2: float, taille_troupeau: int, espece: str, surface_ha: float = 78.5) -> dict:
    """
    Calcule la capacité de charge et le plan de rotation.
    CORRECTION : Le modèle CSIRO prédit des grammes pour un quadrat 0.25m²
    On divise par 0.25 pour obtenir la densité réelle en g/m²
    puis on multiplie par la surface réelle tracée sur la carte.
    """
    consommation_jour = {
        "Vache":   10000,
        "Mouton":  1000,
        "Chèvre":  875,
        "Chameau": 7500,
    }

    # Conversion quadrat → densité g/m²
    # Le modèle prédit des g pour 0.25 m² → on divise par 0.25
    SURFACE_QUADRAT_M2 = 0.25
    densite_g_m2 = gdm_g_par_m2 / SURFACE_QUADRAT_M2

    # Surface réelle tracée sur la carte (m²)
    surface_m2 = surface_ha * 10000

    # Biomasse totale utilisable (50% taux d'utilisation recommandé)
    biomasse_totale_g = densite_g_m2 * surface_m2 * 0.50

    # Consommation du troupeau par jour
    conso_par_animal = consommation_jour.get(espece, 5000)
    conso_troupeau_jour = conso_par_animal * taille_troupeau

    # Nombre de jours disponibles
    if conso_troupeau_jour > 0:
        jours_disponibles = int(biomasse_totale_g / conso_troupeau_jour)
    else:
        jours_disponibles = 0

    # Recommandation
    if jours_disponibles >= 14:
        statut = "excellent"
        couleur = "#22c55e"
        message = f"Excellente zone — votre troupeau peut rester {jours_disponibles} jours"
        action  = "Restez sur cette zone. Planifiez le prochain déplacement dans 10 jours."
    elif jours_disponibles >= 7:
        statut = "bon"
        couleur = "#f59e0b"
        message = f"Bonne zone — {jours_disponibles} jours disponibles"
        action  = f"Préparez le déplacement dans {max(1, jours_disponibles - 3)} jours."
    elif jours_disponibles >= 3:
        statut = "attention"
        couleur = "#f97316"
        message = f"Zone en fin de cycle — seulement {jours_disponibles} jours"
        action  = "Identifiez une nouvelle zone dès aujourd'hui."
    else:
        statut = "critique"
        couleur = "#ef4444"
        message = "Zone épuisée — déplacement immédiat recommandé"
        action  = "Quittez cette zone immédiatement pour éviter la dégradation du sol."

    return {
        "jours_disponibles": jours_disponibles,
        "statut": statut,
        "couleur": couleur,
        "message": message,
        "action": action,
        "biomasse_totale_kg": round(biomasse_totale_g / 1000, 1),
        "surface_ha": round(surface_m2 / 10000, 1),
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def accueil(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyser")
async def analyser_photo(
    file: UploadFile = File(...),
    taille_troupeau: int = 50,
    espece: str = "Vache",
    surface_ha: float = 0.0   # 0 = non fournie, on utilise 78.5 ha par défaut
):
    """
    Reçoit une photo + surface tracée sur la carte, retourne le plan de rotation.
    """
    try:
        contenu = await file.read()
        image = Image.open(io.BytesIO(contenu))
        entree = pretraiter_image(image)
        sorties = session.run(None, {NOM_ENTREE: entree})[0][0]

        green_g  = max(0, float(sorties[0]))
        clover_g = max(0, float(sorties[1]))
        dead_g   = max(0, float(sorties[2]))
        gdm_g    = green_g + clover_g
        total_g  = gdm_g + dead_g
        qualite  = min(100, int((gdm_g / max(total_g, 1)) * 100))

        # Utilise la surface tracée sur la carte si fournie, sinon 78.5 ha par défaut
        surface_reelle_ha = surface_ha if surface_ha > 0 else 78.5

        plan = calculer_plan_rotation(gdm_g, taille_troupeau, espece, surface_reelle_ha)

        return JSONResponse({
            "succes": True,
            "biomasse": {
                "vegetation_verte_g": round(green_g, 1),
                "trefle_g":           round(clover_g, 1),
                "matiere_morte_g":    round(dead_g, 1),
                "gdm_g":              round(gdm_g, 1),
                "total_g":            round(total_g, 1),
                "qualite_pourcent":   qualite,
            },
            "plan": plan,
            "troupeau": {
                "taille": taille_troupeau,
                "espece": espece,
                "surface_ha": round(surface_reelle_ha, 2),
            }
        })

    except Exception as e:
        return JSONResponse({"succes": False, "erreur": str(e)}, status_code=500)


@app.get("/sante")
async def sante():
    return {"statut": "ok", "modele": "nexa_biomasse.onnx", "version": "1.0.0"}