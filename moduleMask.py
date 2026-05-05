"""
moduleMask.py
=============
Module de détection de masque facial utilisant TensorFlow/MobileNetV2.
Usage simple : import moduleMask puis appeler ouvrirCam(), detecterVisage(), detecterMask()

Dépendances :
    pip install opencv-python tensorflow numpy
"""

import cv2
import numpy as np
import urllib.request
import os

# ──────────────────────────────────────────────
# Chargement lazy (une seule fois au démarrage)
# ──────────────────────────────────────────────
_cap        = None   # Objet caméra OpenCV
_face_net   = None   # Réseau de détection de visages (DNN OpenCV)
_mask_model = None   # Modèle TensorFlow de classification masque/sans-masque

# Seuils
SEUIL_DETECTION_VISAGE = 0.5   # Confiance minimum pour valider un visage détecté
SEUIL_MASQUE           = 0.75  # Probabilité minimum pour valider "masque porté"

# URLs des fichiers du modèle de détection de visage (Caffe DNN)
FACE_PROTO_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Chemins locaux des fichiers
FACE_PROTO_PATH = "face_detector.prototxt"
FACE_MODEL_PATH = "face_detector.caffemodel"
MASK_MODEL_PATH = "mask_detector.keras"


# ──────────────────────────────────────────────
# Fonctions internes (privées)
# ──────────────────────────────────────────────

def _telecharger_si_absent(url, chemin):
    """Télécharge un fichier seulement s'il n'existe pas déjà."""
    if not os.path.exists(chemin):
        print(f"[moduleMask] Téléchargement : {chemin} ...")
        urllib.request.urlretrieve(url, chemin)
        print(f"[moduleMask] ✓ {chemin} prêt.")


def _charger_face_net():
    """Charge le réseau de détection de visages OpenCV DNN (Caffe)."""
    global _face_net
    if _face_net is None:
        _telecharger_si_absent(FACE_PROTO_URL,  FACE_PROTO_PATH)
        _telecharger_si_absent(FACE_MODEL_URL,  FACE_MODEL_PATH)
        _face_net = cv2.dnn.readNet(FACE_MODEL_PATH, FACE_PROTO_PATH)
        print("[moduleMask] ✓ Réseau de détection visage chargé.")


def _charger_mask_model():
    """
    Charge le modèle de classification masque.
    Si le modèle local n'existe pas, on l'entraîne rapidement avec MobileNetV2.
    """
    global _mask_model
    if _mask_model is not None:
        return

    # Import TensorFlow ici pour ne pas ralentir l'import du module si non utilisé
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except ImportError:
        raise ImportError(
            "[moduleMask] TensorFlow non installé. Exécutez : pip install tensorflow"
        )

    if os.path.exists(MASK_MODEL_PATH):
        print("[moduleMask] Chargement du modèle masque depuis le disque ...")
        _mask_model = load_model(MASK_MODEL_PATH)
        print("[moduleMask] ✓ Modèle masque chargé.")
    else:
        print("[moduleMask] Modèle non trouvé. Construction du modèle MobileNetV2 ...")
        # Construire l'architecture MobileNetV2 fine-tunée (transfer learning)
        base = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3))
        )
        # Geler la base pour garder les poids ImageNet
        base.trainable = False

        head = base.output
        head = AveragePooling2D(pool_size=(7, 7))(head)
        head = Flatten()(head)
        head = Dense(128, activation="relu")(head)
        head = Dropout(0.5)(head)
        head = Dense(2, activation="softmax")(head)  # [sans_masque, avec_masque]

        _mask_model = Model(inputs=base.input, outputs=head)
        _mask_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        _mask_model.save(MASK_MODEL_PATH)
        print("[moduleMask] ✓ Modèle créé et sauvegardé (non entraîné sur données réelles).")
        print("[moduleMask] ⚠ Pour une détection fiable, entraîner avec un dataset masque.")


def _detecter_visages_frame(frame):
    """
    Détecte les visages dans une frame OpenCV.
    Retourne une liste de boîtes (startX, startY, endX, endY).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    _face_net.setInput(blob)
    detections = _face_net.forward()

    visages = []
    for i in range(detections.shape[2]):
        confiance = detections[0, 0, i, 2]
        if confiance < SEUIL_DETECTION_VISAGE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        # Sécuriser les coordonnées
        startX = max(0, startX)
        startY = max(0, startY)
        endX   = min(w - 1, endX)
        endY   = min(h - 1, endY)
        visages.append((startX, startY, endX, endY))
    return visages


def _classifier_masque(frame, visages):
    """
    Pour chaque visage détecté, prédit si la personne porte un masque.
    Retourne une liste de dicts : {box, masque_porte, proba_masque}
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    resultats = []
    for (startX, startY, endX, endY) in visages:
        roi = frame[startY:endY, startX:endX]
        if roi.size == 0:
            continue
        roi_rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resize = cv2.resize(roi_rgb, (224, 224))
        roi_array  = preprocess_input(np.expand_dims(roi_resize, axis=0))

        (sans_masque, avec_masque) = _mask_model.predict(roi_array, verbose=0)[0]
        masque_porte = avec_masque > SEUIL_MASQUE

        resultats.append({
            "box":          (startX, startY, endX, endY),
            "masque_porte": masque_porte,
            "proba_masque": float(avec_masque),
        })
    return resultats


def _dessiner_resultats(frame, resultats):
    """Dessine les boîtes et labels sur la frame."""
    for r in resultats:
        (startX, startY, endX, endY) = r["box"]
        if r["masque_porte"]:
            couleur = (0, 255, 0)   # Vert
            label   = f"MASQUE  {r['proba_masque']*100:.0f}%"
        else:
            couleur = (0, 0, 255)   # Rouge
            label   = f"SANS MASQUE  {(1 - r['proba_masque'])*100:.0f}%"

        cv2.rectangle(frame, (startX, startY), (endX, endY), couleur, 2)
        y_label = startY - 10 if startY - 10 > 10 else startY + 20
        cv2.putText(frame, label, (startX, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, couleur, 2)
    return frame


# ──────────────────────────────────────────────
# API publique du module
# ──────────────────────────────────────────────

def ouvrirCam(index=0):
    """
    Ouvre la caméra.
    :param index: Index de la caméra (0 = caméra par défaut)
    """
    global _cap
    if _cap is not None and _cap.isOpened():
        print("[moduleMask] Caméra déjà ouverte.")
        return

    _cap = cv2.VideoCapture(index)
    if not _cap.isOpened():
        raise RuntimeError(f"[moduleMask] Impossible d'ouvrir la caméra {index}.")

    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[moduleMask] ✓ Caméra {index} ouverte (640×480).")
    _charger_face_net()


def fermerCam():
    """Libère la caméra proprement."""
    global _cap
    if _cap:
        _cap.release()
        _cap = None
        cv2.destroyAllWindows()
        print("[moduleMask] Caméra fermée.")


def lireFrame():
    """
    Lit une frame depuis la caméra.
    :return: frame BGR ou None si échec
    """
    if _cap is None or not _cap.isOpened():
        raise RuntimeError("[moduleMask] Caméra non ouverte. Appeler ouvrirCam() d'abord.")
    ret, frame = _cap.read()
    return frame if ret else None


def detecterVisage(afficher=True):
    """
    Lit une frame et détecte les visages.
    :param afficher: Affiche la frame avec les rectangles si True
    :return: liste de boîtes (startX, startY, endX, endY)
    """
    frame = lireFrame()
    if frame is None:
        return []

    visages = _detecter_visages_frame(frame)

    if afficher:
        for (startX, startY, endX, endY) in visages:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 165, 0), 2)
        cv2.putText(frame, f"Visages : {len(visages)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.imshow("Detection Visage", frame)
        cv2.waitKey(1)

    return visages


def detecterMask(afficher=True):
    """
    Lit une frame, détecte les visages et classe masque/sans-masque.
    :param afficher: Affiche la frame annotée si True
    :return: dict avec :
             - "visages_detectes" : int
             - "tous_avec_masque" : bool  (True si tous portent un masque)
             - "resultats"        : liste de dicts par visage
    """
    _charger_mask_model()

    frame = lireFrame()
    if frame is None:
        return {"visages_detectes": 0, "tous_avec_masque": False, "resultats": []}

    visages   = _detecter_visages_frame(frame)
    resultats = _classifier_masque(frame, visages)

    tous_avec_masque = (
        len(resultats) > 0 and all(r["masque_porte"] for r in resultats)
    )

    if afficher:
        frame = _dessiner_resultats(frame, resultats)
        # Bannière de statut global
        if len(resultats) == 0:
            texte_global, coul_global = "Aucun visage", (200, 200, 200)
        elif tous_avec_masque:
            texte_global, coul_global = "ACCES AUTORISE", (0, 255, 0)
        else:
            texte_global, coul_global = "ACCES REFUSE - PAS DE MASQUE", (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (640, 45), (30, 30, 30), -1)
        cv2.putText(frame, texte_global, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, coul_global, 2)
        cv2.imshow("Detection Masque", frame)
        cv2.waitKey(1)

    return {
        "visages_detectes": len(resultats),
        "tous_avec_masque": tous_avec_masque,
        "resultats":        resultats,
    }
