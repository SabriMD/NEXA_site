# ============================================================
# NEXA — Conversion du modèle .pth → ONNX → TFLite
# À exécuter localement sur ton PC après téléchargement du .pth
# ============================================================
# Installation requise :
#   pip install torch torchvision timm onnx onnxruntime
#   pip install onnx-tf tensorflow
# ============================================================
import onnx
import onnxruntime as ort
import torch
import timm
import numpy as np
import os

# ============================================================
# CONFIGURATION — Modifie ce chemin selon ton fichier téléchargé
# ============================================================
CHEMIN_MODELE = 'modeles/meilleur_modele_fold2.pth'
NOM_ONNX      = 'modeles/nexa_biomasse.onnx'
NOM_MODELE    = 'efficientnet_b2'
TAILLE_IMAGE  = (260, 260)  # Taille d'entrée EfficientNet-B2
NOM_TFLITE    = 'nexa_biomasse.tflite'

# ============================================================
# ÉTAPE 1 — Chargement du modèle PyTorch
# ============================================================
print("📂 Chargement du modèle PyTorch...")

model = timm.create_model(NOM_MODELE, pretrained=False, num_classes=3)
model.load_state_dict(
    torch.load(CHEMIN_MODELE, map_location='cpu', weights_only=True)
)
model.eval()
print(f"✅ Modèle chargé : {CHEMIN_MODELE}")

# ============================================================
# ÉTAPE 2 — Test rapide : vérifier que le modèle tourne
# ============================================================
print("\n🧪 Test du modèle sur une image factice...")
image_factice = torch.randn(1, 3, TAILLE_IMAGE[0], TAILLE_IMAGE[1])
with torch.no_grad():
    sortie = model(image_factice)

print(f"✅ Sortie du modèle : {sortie.numpy()}")
print(f"   → Green: {sortie[0][0]:.2f}g | Clover: {sortie[0][1]:.2f}g | Dead: {sortie[0][2]:.2f}g")
gdm   = sortie[0][0].item() + sortie[0][1].item()
total = gdm + sortie[0][2].item()
print(f"   → GDM: {gdm:.2f}g | Total: {total:.2f}g")

# ============================================================
# ÉTAPE 3 — Export vers ONNX
# ============================================================
print(f"\n📦 Export vers ONNX → {NOM_ONNX}")

torch.onnx.export(
    model,
    image_factice,
    NOM_ONNX,
    input_names=["image"],
    output_names=["biomasse"],
    dynamic_axes={
        "image":    {0: "batch_size"},
        "biomasse": {0: "batch_size"}
    },
    opset_version=11,
    export_params=True,
    do_constant_folding=True,  # Optimisation du graphe
)
print(f"✅ Fichier ONNX généré : {NOM_ONNX} ({os.path.getsize(NOM_ONNX) / 1e6:.1f} MB)")

# ============================================================
# ÉTAPE 4 — Validation du modèle ONNX
# ============================================================
print("\n🔍 Validation du modèle ONNX...")

modele_onnx = onnx.load(NOM_ONNX)
onnx.checker.check_model(modele_onnx)
print("✅ Modèle ONNX valide")

# Test d'inférence ONNX

session = ort.InferenceSession(NOM_ONNX)
entree = {session.get_inputs()[0].name: image_factice.numpy()}
sortie_onnx = session.run(None, entree)[0]
print(f"✅ Inférence ONNX OK : {sortie_onnx}")

# Vérification cohérence PyTorch vs ONNX
diff = np.abs(sortie.numpy() - sortie_onnx).max()
print(f"   Différence max PyTorch vs ONNX : {diff:.6f} (doit être < 0.001)")
assert diff < 0.01, f"⚠️ Différence trop grande : {diff}"

# ============================================================
# ÉTAPE 5 — Conversion ONNX → TFLite
# ============================================================
print(f"\n📱 Conversion vers TFLite → {NOM_TFLITE}")
print("(Cette étape peut prendre 2-5 minutes...)")

try:
    # Méthode recommandée via onnx-tf
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # ONNX → TensorFlow SavedModel
    DOSSIER_TF = 'nexa_biomasse_tf_savedmodel'
    print("  → Conversion ONNX vers TensorFlow SavedModel...")
    tf_rep = prepare(modele_onnx)
    tf_rep.export_graph(DOSSIER_TF)
    print(f"  ✅ SavedModel TF généré dans : {DOSSIER_TF}/")

    # TensorFlow SavedModel → TFLite
    print("  → Conversion SavedModel vers TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(DOSSIER_TF)

    # Optimisation pour mobile (réduit la taille du modèle)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(NOM_TFLITE, 'wb') as f:
        f.write(tflite_model)

    taille_mb = os.path.getsize(NOM_TFLITE) / 1e6
    print(f"  ✅ Fichier TFLite généré : {NOM_TFLITE} ({taille_mb:.1f} MB)")

    # ============================================================
    # ÉTAPE 6 — Validation du modèle TFLite
    # ============================================================
    print("\n🔍 Validation du modèle TFLite...")
    interpreteur = tf.lite.Interpreter(model_path=NOM_TFLITE)
    interpreteur.allocate_tensors()

    infos_entree  = interpreteur.get_input_details()
    infos_sortie  = interpreteur.get_output_details()

    print(f"  Entrée  : shape={infos_entree[0]['shape']}  dtype={infos_entree[0]['dtype']}")
    print(f"  Sortie  : shape={infos_sortie[0]['shape']} dtype={infos_sortie[0]['dtype']}")

    # Test d'inférence TFLite
    interpreteur.set_tensor(infos_entree[0]['index'], image_factice.numpy())
    interpreteur.invoke()
    sortie_tflite = interpreteur.get_tensor(infos_sortie[0]['index'])
    print(f"  ✅ Inférence TFLite OK : {sortie_tflite}")

    diff_tflite = np.abs(sortie.numpy() - sortie_tflite).max()
    print(f"  Différence max PyTorch vs TFLite : {diff_tflite:.6f}")

except ImportError:
    print("\n⚠️  onnx-tf ou tensorflow non installé.")
    print("Lance ces commandes d'abord :")
    print("  pip install onnx-tf tensorflow")
    print("\nOu utilise directement le fichier ONNX avec ONNX Runtime sur Android.")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print("\n" + "="*50)
print("🎯 RÉSUMÉ DES FICHIERS GÉNÉRÉS")
print("="*50)

fichiers = [NOM_ONNX, NOM_TFLITE]
for f in fichiers:
    if os.path.exists(f):
        print(f"  ✅ {f} ({os.path.getsize(f) / 1e6:.1f} MB)")
    else:
        print(f"  ❌ {f} — non généré")

print("\n📱 PROCHAINE ÉTAPE :")
print(f"  Copie '{NOM_TFLITE}' dans le dossier assets/ de ton app Android/React Native")
print("  Utilise react-native-fast-tflite pour l'inférence offline")