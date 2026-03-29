"""
main.py — Backend FastAPI · NeuralBase ML Studio
=================================================
API REST pour tester les modèles entraînés depuis neuralbase.html

Démarrage local :
    uvicorn main:app --reload --port 8000

Déploiement Render :
    Start Command → uvicorn main:app --host 0.0.0.0 --port $PORT
    Build Command → pip install -r requirements.txt
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTIF CRITIQUE : importer les classes personnalisées AVANT load_model()
# Cela déclenche @register_keras_serializable() et enregistre les classes
# dans le registre interne de Keras.
# ══════════════════════════════════════════════════════════════════════════════
from models.cnn_model import CustomCNN        # noqa: F401
from models.rnn_model import LSTMForecaster   # noqa: F401


# ══════════════════════════════════════════════════════════════════════════════
# Logging structuré
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuralbase")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CNN_PATH  = os.getenv("CNN_MODEL_PATH",  "models/saved/cnn_model.keras")
LSTM_PATH = os.getenv("LSTM_MODEL_PATH", "models/saved/rnn_model.keras")

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_MB", 10))
MAX_IMAGE_BYTES   = MAX_IMAGE_SIZE_MB * 1024 * 1024

CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]


# ══════════════════════════════════════════════════════════════════════════════
# État global des modèles
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    cnn:  Optional[tf.keras.Model] = None
    lstm: Optional[tf.keras.Model] = None
    cnn_load_error:  str = ""
    lstm_load_error: str = ""
    startup_time: float = 0.0


registry = ModelRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage ; libère les ressources à l'arrêt."""
    t0 = time.perf_counter()
    logger.info("⚡ Chargement des modèles…")

    # CNN
    if os.path.exists(CNN_PATH):
        try:
            registry.cnn = tf.keras.models.load_model(CNN_PATH)
            logger.info("✅ CNN chargé : %s", CNN_PATH)
        except Exception as exc:
            registry.cnn_load_error = str(exc)
            logger.error("❌ Échec chargement CNN : %s", exc)
    else:
        registry.cnn_load_error = f"Fichier introuvable : {CNN_PATH}"
        logger.warning("⚠️  CNN non trouvé → lancez train.py --mission cnn")

    # LSTM
    if os.path.exists(LSTM_PATH):
        try:
            registry.lstm = tf.keras.models.load_model(LSTM_PATH)
            logger.info("✅ LSTM chargé : %s", LSTM_PATH)
        except Exception as exc:
            registry.lstm_load_error = str(exc)
            logger.error("❌ Échec chargement LSTM : %s", exc)
    else:
        registry.lstm_load_error = f"Fichier introuvable : {LSTM_PATH}"
        logger.warning("⚠️  LSTM non trouvé → lancez train.py --mission lstm")

    registry.startup_time = round(time.perf_counter() - t0, 3)
    logger.info("🚀 API prête en %.3f s", registry.startup_time)

    yield

    logger.info("🛑 Arrêt de l'API")
    tf.keras.backend.clear_session()


# ══════════════════════════════════════════════════════════════════════════════
# Application FastAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NeuralBase · ML Studio API",
    description=(
        "Backend FastAPI pour **NeuralBase** — évaluation de modèles CNN & LSTM.\n\n"
        "- **`POST /predict/cnn`** — classification d'images CIFAR-10\n"
        "- **`POST /predict/lstm`** — prédiction de série temporelle T+1\n"
        "- **`GET  /health`** — statut des modèles"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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
    duration = (time.perf_counter() - t_start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Erreur non gérée sur %s : %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Erreur interne du serveur.", "data": {}},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Schémas Pydantic
# ══════════════════════════════════════════════════════════════════════════════

class LSTMRequest(BaseModel):
    sequence: list[float] = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Fenêtre temporelle de N valeurs (min 2, max 500).",
        examples=[[0.12, 0.15, 0.18, 0.21, 0.24]],
    )

    @field_validator("sequence")
    @classmethod
    def no_nan_inf(cls, values: list[float]) -> list[float]:
        if any(not np.isfinite(v) for v in values):
            raise ValueError("La séquence ne doit pas contenir NaN ou Inf.")
        return values


class Top3Item(BaseModel):
    class_name: str = Field(alias="class")
    probability: float
    model_config = {"populate_by_name": True}


class CNNData(BaseModel):
    predicted_class: str
    confidence: float
    top3: list[Top3Item]
    all_probabilities: dict[str, float]


class LSTMData(BaseModel):
    prediction_t_plus_1: float
    sequence_length_used: int


class APIResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Statut de l'API", tags=["Monitoring"])
async def health_check():
    return {
        "status": "ok",
        "version": app.version,
        "cnn_loaded":      registry.cnn is not None,
        "lstm_loaded":     registry.lstm is not None,
        "cnn_load_error":  registry.cnn_load_error or None,
        "lstm_load_error": registry.lstm_load_error or None,
        "startup_time_s":  registry.startup_time,
        "tf_version":      tf.__version__,
    }


@app.post(
    "/predict/cnn",
    response_model=APIResponse,
    summary="Classification d'image (CIFAR-10)",
    tags=["Modèles"],
)
async def predict_image(file: UploadFile = File(..., description="Image JPEG/PNG/WebP")):
    if registry.cnn is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_UNAVAILABLE",
                "message": "Modèle CNN non disponible.",
                "hint": "Lancez : python train.py --mission cnn",
                "load_error": registry.cnn_load_error,
            },
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail={"error": "UNSUPPORTED_MEDIA", "message": "Seules les images sont acceptées (image/*)"},
        )

    contents = await file.read()

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "FILE_TOO_LARGE",
                "message": f"Taille max : {MAX_IMAGE_SIZE_MB} MB. Reçu : {len(contents) / 1024 / 1024:.1f} MB",
            },
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32), Image.LANCZOS)
        img_array = np.array(image, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "IMAGE_PROCESSING_ERROR", "message": f"Impossible de traiter l'image : {exc}"},
        ) from exc

    try:
        probabilities: np.ndarray = registry.cnn.predict(img_array, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "INFERENCE_ERROR", "message": f"Erreur d'inférence CNN : {exc}"},
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

    return APIResponse(
        success=True,
        message=f"Classe prédite : {predicted_class} ({confidence * 100:.1f}%)",
        data={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top3": top3,
            "all_probabilities": all_probs,
        },
    )


@app.post(
    "/predict/lstm",
    response_model=APIResponse,
    summary="Prédiction série temporelle (T+1)",
    tags=["Modèles"],
)
async def predict_timeseries(request: LSTMRequest):
    if registry.lstm is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_UNAVAILABLE",
                "message": "Modèle LSTM non disponible.",
                "hint": "Lancez : python train.py --mission lstm",
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
            detail={"error": "INFERENCE_ERROR", "message": f"Erreur d'inférence LSTM : {exc}"},
        ) from exc

    return APIResponse(
        success=True,
        message=f"Prédiction T+1 : {pred_value:.4f}",
        data={
            "prediction_t_plus_1": round(pred_value, 6),
            "sequence_length_used": len(request.sequence),
        },
    )
