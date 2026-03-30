"""
main.py — Backend FastAPI · NeuralBase ML Studio  v2
=====================================================
Démarrage local :
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, field_validator

try:
    from models.cnn_model import CustomCNN, ResBlock
except ImportError:
    from models.cnn_model import CustomCNN
    ResBlock = None
from models.rnn_model import LSTMForecaster 

# Dictionnaire passé à load_model() pour désérialiser les classes custom
CUSTOM_OBJECTS = {
    "CustomCNN":      CustomCNN,
    "ResBlock":       ResBlock,
    "LSTMForecaster": LSTMForecaster,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuralbase")


CNN_PATH  = os.getenv("CNN_MODEL_PATH",  "models/saved/cnn_model.keras")
LSTM_PATH = os.getenv("LSTM_MODEL_PATH", "models/saved/rnn_model.keras")

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_MB", 10))
MAX_IMAGE_BYTES   = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Seuil de confiance minimum en dessous, on avertit l'utilisateur
CONFIDENCE_WARNING_THRESHOLD = 0.40

CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]

# Classes CIFAR-10 qui existent dans la vie réelle en photos haute résolution
PHOTO_FRIENDLY_CLASSES = {"chat", "chien", "cheval", "cerf", "oiseau"}

# Message important sur les limitations du modèle
CIFAR10_LIMITATION_MSG = (
    "CIFAR-10 ne contient PAS de classe 'humain/personne'. "
    "Le modèle est entraîné sur des images 32×32 très basse résolution. "
    "Pour les photos réelles, la confiance sera souvent faible."
)


def resolve_model_path(primary: str) -> tuple[str | None, str]:
    """
    Cherche le modèle dans l'ordre :
      1. primary            (chemin configuré via env ou défaut)
      2. extension basculée (.keras ↔ .h5)
    """
    if os.path.exists(primary):
        return primary, f"Chargé depuis : {primary}"

    # Bascule d'extension
    if primary.endswith(".keras"):
        fallback = primary[:-6] + ".h5"
    elif primary.endswith(".h5"):
        fallback = primary[:-3] + ".keras"
    else:
        fallback = None

    if fallback and os.path.exists(fallback):
        return fallback, f"'{primary}' absent → fallback utilisé : '{fallback}'"

    tried = [primary] + ([fallback] if fallback else [])
    return None, "Fichier introuvable parmi : " + ", ".join(tried)

class ModelRegistry:
    cnn:  Optional[tf.keras.Model] = None
    lstm: Optional[tf.keras.Model] = None
    cnn_load_error:  str = ""
    lstm_load_error: str = ""
    cnn_resolved_path:  str = ""
    lstm_resolved_path: str = ""
    startup_time: float = 0.0


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    logger.info("⚡ Chargement des modèles…")

    #CNN
    cnn_path, cnn_msg = resolve_model_path(CNN_PATH)
    if cnn_path:
        try:
            registry.cnn = tf.keras.models.load_model(
                cnn_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
            registry.cnn_resolved_path = cnn_path
            logger.info("CNN chargé : %s", cnn_msg)
        except Exception as exc:
            registry.cnn_load_error = str(exc)
            logger.error("Échec CNN (%s) : %s", cnn_path, exc)
    else:
        registry.cnn_load_error = cnn_msg
        logger.warning("CNN absent → %s | Lancez : python train.py --mission cnn", cnn_msg)

    # LSTM
    lstm_path, lstm_msg = resolve_model_path(LSTM_PATH)
    if lstm_path:
        try:
            registry.lstm = tf.keras.models.load_model(
                lstm_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
            registry.lstm_resolved_path = lstm_path
            logger.info("LSTM chargé : %s", lstm_msg)
        except Exception as exc:
            registry.lstm_load_error = str(exc)
            logger.error("Échec LSTM (%s) : %s", lstm_path, exc)
    else:
        registry.lstm_load_error = lstm_msg
        logger.warning("LSTM absent → %s | Lancez : python train.py --mission lstm", lstm_msg)

    registry.startup_time = round(time.perf_counter() - t0, 3)
    logger.info("API prête en %.3f s", registry.startup_time)
    yield
    logger.info("Arrêt de l'API")
    tf.keras.backend.clear_session()

# Application
app = FastAPI(
    title="NeuralBase · ML Studio API v2",
    description=(
        "Backend FastAPI pour **NeuralBase** — CNN & LSTM.\n\n"
        "- **`POST /predict/cnn`** — classification d'images CIFAR-10\n"
        "- **`POST /predict/lstm`** — prédiction série temporelle T+1\n"
        "- **`GET  /health`** — statut des modèles\n"
        "- **`GET  /model/info`** — informations détaillées\n\n"
        f"> {CIFAR10_LIMITATION_MSG}"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t_start = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code,
        (time.perf_counter() - t_start) * 1000,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Erreur non gérée sur %s : %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Erreur interne du serveur.", "data": {}},
    )

# Schémas Pydantic
class LSTMRequest(BaseModel):
    sequence: List[float] = Field(
        ..., min_length=2, max_length=500,
        description="Fenêtre temporelle N valeurs (2–500).",
        examples=[[0.12, 0.15, 0.18, 0.21, 0.24]],
    )

    @field_validator("sequence")
    @classmethod
    def no_nan_inf(cls, values: List[float]) -> List[float]:
        if any(not np.isfinite(v) for v in values):
            raise ValueError("La séquence ne doit pas contenir NaN ou Inf.")
        return values


class APIResponse(BaseModel):
    success: bool
    message: str = ""
    warning: Optional[str] = None
    data: Dict = {}

# Preprocessing image
def preprocess_image_for_cifar(image_bytes: bytes) -> np.ndarray:
    """
    Prétraitement robuste pour CIFAR-10 :
      1. Ouvre et convertit en RGB
      2. Centre-crop carré (évite la déformation)
      3. Redimensionne en 32×32 avec LANCZOS
      4. Normalise [0, 1]
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Centre-crop carré avant resize (préserve les proportions)
    w, h = image.size
    min_side = min(w, h)
    left   = (w - min_side) // 2
    top    = (h - min_side) // 2
    right  = left + min_side
    bottom = top  + min_side
    image  = image.crop((left, top, right, bottom))

    image = image.resize((32, 32), Image.LANCZOS)
    img_array = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(img_array, axis=0)   # shape (1, 32, 32, 3)


def build_warning(confidence: float, predicted_class: str) -> Optional[str]:
    # Construit un message d'avertissement contextuel si nécessaire. Retourne None si tout va bien.
    warnings = []

    if confidence < CONFIDENCE_WARNING_THRESHOLD:
        warnings.append(
            f"Confiance faible ({confidence*100:.1f}%). "
            "L'image est peut-être hors-distribution CIFAR-10 "
            "(ex: portrait, paysage, objet non listé)."
        )

    warnings.append(
        "CIFAR-10 reconnaît uniquement : avion, automobile, oiseau, "
        "chat, cerf, chien, grenouille, cheval, bateau, camion. "
        "Les humains/personnes ne font PAS partie des classes entraînées."
    )

    return " | ".join(warnings) if warnings else None

# Routes
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Statut de l'API", tags=["Monitoring"])
async def health_check():
    return {
        "status":               "ok",
        "version":              app.version,
        "cnn_loaded":           registry.cnn is not None,
        "lstm_loaded":          registry.lstm is not None,
        "cnn_resolved_path":    registry.cnn_resolved_path or None,
        "lstm_resolved_path":   registry.lstm_resolved_path or None,
        "cnn_load_error":       registry.cnn_load_error or None,
        "lstm_load_error":      registry.lstm_load_error or None,
        "startup_time_s":       registry.startup_time,
        "tf_version":           tf.__version__,
        "cifar10_classes":      CIFAR10_CLASSES,
        "important_note":       CIFAR10_LIMITATION_MSG,
    }


@app.get("/model/info", summary="Informations sur les modèles", tags=["Monitoring"])
async def model_info():
    """Retourne les métadonnées complètes des modèles chargés."""
    cnn_params  = None
    lstm_params = None

    if registry.cnn:
        try:
            cnn_params = registry.cnn.count_params()
        except Exception:
            pass

    if registry.lstm:
        try:
            lstm_params = registry.lstm.count_params()
        except Exception:
            pass

    return {
        "cnn": {
            "loaded":           registry.cnn is not None,
            "path_configured":  CNN_PATH,
            "path_resolved":    registry.cnn_resolved_path or None,
            "parameters":       cnn_params,
            "input_shape":      "32x32x3",
            "output_classes":   10,
            "classes":          CIFAR10_CLASSES,
            "architecture":     "ResNet-like (blocs résiduels) v2",
            "important_note":   CIFAR10_LIMITATION_MSG,
        },
        "lstm": {
            "loaded":           registry.lstm is not None,
            "path_configured":  LSTM_PATH,
            "path_resolved":    registry.lstm_resolved_path or None,
            "parameters":       lstm_params,
            "task":             "Prédiction série temporelle T+1",
            "dataset":          "Jena Climate 2009-2016",
        },
    }


@app.post(
    "/predict/cnn",
    response_model=APIResponse,
    summary="Classification d'image CIFAR-10",
    tags=["Modèles"],
)
async def predict_image(file: UploadFile = File(..., description="Image JPEG/PNG/WebP")):
    if registry.cnn is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error":      "MODEL_UNAVAILABLE",
                "message":    "Modèle CNN non disponible.",
                "hint":       "Lancez : python train.py --mission cnn",
                "load_error": registry.cnn_load_error,
            },
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail={"error": "UNSUPPORTED_MEDIA", "message": "Seules les images sont acceptées."},
        )

    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error":   "FILE_TOO_LARGE",
                "message": f"Max {MAX_IMAGE_SIZE_MB} MB. Reçu : {len(contents)/1024/1024:.1f} MB",
            },
        )

    try:
        img_array = preprocess_image_for_cifar(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "IMAGE_PROCESSING_ERROR", "message": str(exc)},
        ) from exc

    try:
        probabilities: np.ndarray = registry.cnn.predict(img_array, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "INFERENCE_ERROR", "message": str(exc)},
        ) from exc

    predicted_idx   = int(np.argmax(probabilities))
    predicted_class = CIFAR10_CLASSES[predicted_idx]
    confidence      = float(probabilities[predicted_idx])

    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {"class": CIFAR10_CLASSES[i], "probability": round(float(probabilities[i]), 4)}
        for i in top3_idx
    ]
    all_probs = {
        CIFAR10_CLASSES[i]: round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    warning = build_warning(confidence, predicted_class)

    return APIResponse(
        success=True,
        message=f"Classe prédite : {predicted_class} ({confidence * 100:.1f}%)",
        warning=warning,
        data={
            "predicted_class":    predicted_class,
            "confidence":         round(confidence, 4),
            "top3":               top3,
            "all_probabilities":  all_probs,
            "cifar10_classes":    CIFAR10_CLASSES,
            "confidence_level":   (
                "haute"   if confidence >= 0.70 else
                "moyenne" if confidence >= 0.40 else
                "faible"
            ),
        },
    )


@app.post(
    "/predict/lstm",
    response_model=APIResponse,
    summary="Prédiction série temporelle T+1",
    tags=["Modèles"],
)
async def predict_timeseries(request: LSTMRequest):
    """Prédit la valeur T+1 à partir d'une séquence temporelle normalisée."""
    if registry.lstm is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error":      "MODEL_UNAVAILABLE",
                "message":    "Modèle LSTM non disponible.",
                "hint":       "Lancez : python train.py --mission lstm",
                "load_error": registry.lstm_load_error,
            },
        )

    try:
        input_array = np.array(request.sequence, dtype="float32").reshape(
            1, len(request.sequence), 1
        )
        prediction = registry.lstm.predict(input_array, verbose=0)
        pred_value = float(prediction[0][0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "INFERENCE_ERROR", "message": str(exc)},
        ) from exc

    # Avertissement si valeur prédite hors plage [0,1]
    warning = None
    if not (0.0 <= pred_value <= 1.0):
        warning = (
            f"La valeur prédite ({pred_value:.4f}) est hors de [0, 1]. "
            "Vérifiez que votre séquence est normalisée (MinMaxScaler [0,1])."
        )

    return APIResponse(
        success=True,
        message=f"Prédiction T+1 : {pred_value:.4f}",
        warning=warning,
        data={
            "prediction_t_plus_1":  round(pred_value, 6),
            "sequence_length_used": len(request.sequence),
        },
    )
