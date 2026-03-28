"""
main.py — Backend FastAPI
API REST pour tester les modèles entraînés depuis l'interface neuralbase.html

Lancement :
  uvicorn main:app --reload --port 8000

Endpoints :
  POST /predict/cnn   — Upload d'image → classe CIFAR-10 + confiance
  POST /predict/lstm  — Séquence de flottants → valeur T+1
  GET  /health        — Statut de l'API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ── Initialisation ───────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Studio — API d'Évaluation",
    description="API pour tester les modèles CNN et LSTM du projet Deep Learning",
    version="1.0.0",
)

# ── CORS (nécessaire pour appels depuis neuralbase.html) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Classes CIFAR-10 ─────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

# ── Chargement des modèles au démarrage ──────────────────────────────────────
cnn_model  = None
lstm_model = None

CNN_PATH  = "models/saved/cnn_model.keras"
LSTM_PATH = "models/saved/rnn_model.keras"


@app.on_event("startup")
async def load_models():
    global cnn_model, lstm_model

    if os.path.exists(CNN_PATH):
        try:
            cnn_model = tf.keras.models.load_model(CNN_PATH)
            print(f"[✓] Modèle CNN chargé depuis {CNN_PATH}")
        except Exception as e:
            print(f"[✗] Impossible de charger le CNN : {e}")
    else:
        print(f"[!] CNN non trouvé à {CNN_PATH} — lancez train.py d'abord.")

    if os.path.exists(LSTM_PATH):
        try:
            lstm_model = tf.keras.models.load_model(LSTM_PATH)
            print(f"[✓] Modèle LSTM chargé depuis {LSTM_PATH}")
        except Exception as e:
            print(f"[✗] Impossible de charger le LSTM : {e}")
    else:
        print(f"[!] LSTM non trouvé à {LSTM_PATH} — lancez train.py d'abord.")


# ════════════════════════════════════════════════════════════════════════════
# Schémas Pydantic
# ════════════════════════════════════════════════════════════════════════════

class LSTMRequest(BaseModel):
    sequence: list[float]   # Fenêtre de N valeurs normalisées ou brutes


class PredictionResponse(BaseModel):
    success: bool
    data: dict
    message: str = ""


# ════════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et des modèles chargés."""
    return {
        "status": "ok",
        "cnn_loaded":  cnn_model  is not None,
        "lstm_loaded": lstm_model is not None,
    }


@app.post("/predict/cnn", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Classifie une image via le modèle CNN (CIFAR-10).

    - Accepte tout format d'image (JPEG, PNG, etc.)
    - Redimensionne automatiquement en 32×32
    - Retourne la classe prédite + les 3 meilleures probabilités
    """
    if cnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle CNN non disponible. Lancez train.py --mission cnn d'abord."
        )

    # ── Lecture et prétraitement de l'image ──────────────────────────────
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32), Image.LANCZOS)

        # Normalisation [0, 1] — identique au pipeline d'entraînement
        img_array = np.array(image, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)   # shape (1, 32, 32, 3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture image : {str(e)}")

    # ── Inférence ────────────────────────────────────────────────────────
    try:
        probabilities = cnn_model.predict(img_array, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence CNN : {str(e)}")

    predicted_idx   = int(np.argmax(probabilities))
    predicted_class = CIFAR10_CLASSES[predicted_idx]
    confidence      = float(probabilities[predicted_idx])

    # Top-3 prédictions
    top3_idx  = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {"class": CIFAR10_CLASSES[i], "probability": round(float(probabilities[i]), 4)}
        for i in top3_idx
    ]

    return PredictionResponse(
        success=True,
        data={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top3": top3,
            "all_probabilities": {
                CIFAR10_CLASSES[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        },
        message=f"Classe prédite : {predicted_class} ({confidence * 100:.1f}%)"
    )


@app.post("/predict/lstm", response_model=PredictionResponse)
async def predict_timeseries(request: LSTMRequest):
    """
    Prédit la valeur T+1 à partir d'une séquence temporelle.

    Body JSON attendu :
      { "sequence": [v1, v2, ..., v24] }

    La séquence doit être pré-normalisée (MinMaxScaler) côté client,
    ou vous pouvez envoyer les valeurs brutes si vous gérez la normalisation ici.
    """
    if lstm_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle LSTM non disponible. Lancez train.py --mission lstm d'abord."
        )

    sequence = request.sequence

    if len(sequence) < 2:
        raise HTTPException(
            status_code=400,
            detail="La séquence doit contenir au moins 2 valeurs."
        )

    try:
        # Transformation → (1, sequence_length, 1)
        input_array = np.array(sequence, dtype="float32").reshape(1, len(sequence), 1)

        prediction = lstm_model.predict(input_array, verbose=0)
        pred_value = float(prediction[0][0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence LSTM : {str(e)}")

    return PredictionResponse(
        success=True,
        data={
            "prediction_t_plus_1": round(pred_value, 6),
            "sequence_length_used": len(sequence),
        },
        message=f"Prédiction T+1 : {pred_value:.4f}"
    )
