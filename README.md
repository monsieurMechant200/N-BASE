 # NeuralBase — ML Studio
### Deep Learning & Reseaux de neurones· Niveau 3 SDIA | Dr. Noulapeu

> Projet d'évaluation pratique combinant **classification d'images** (CNN / CIFAR-10) et **prédiction de séries temporelles** (LSTM / Jena Climate), exposés via une API REST FastAPI et un dashboard interactif.

---

## Résultats clés

| Mission | Métrique | Valeur | Objectif |
|---------|----------|--------|----------|
| CNN — Classification CIFAR-10 | Test Accuracy | **79.52 %** | > 70 %  |
| LSTM — Prévision Jena Climate | Test MSE | **0.1040** | Minimiser  |

---

## Structure du projet

```
deep_learning_project/
 ├── data/
 │   └── jena_climate_2009_2016.csv
 ├── models/
 │   ├── cnn_model.py          ← Architecture CustomCNN (API Subclassing)
 │   ├── rnn_model.py          ← Architecture LSTMForecaster
 │   └── saved/
 │       ├── cnn_model.keras
 │       └── rnn_model.keras
 ├── utils/
 │   ├── data_processing.py    ← Pipelines CIFAR-10 & Jena Climate
 │   └── visualization.py      ← Graphiques Matplotlib
 ├── results/                  ← PNG générés par evaluate.py
 ├── train.py                  ← Script d'entraînement
 ├── evaluate.py               ← Script d'évaluation
 ├── main.py                   ← Backend FastAPI
 ├── neuralbase.html           ← Dashboard interactif
 ├── README.md
 └── requirements.txt
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/monsieurMechant200/N-BASE.git
cd N-BASE

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**
`tensorflow >= 2.13` · `fastapi` · `uvicorn` · `pillow` · `scikit-learn` · `pandas`

---

## Utilisation

### 1 · Entraîner les modèles

```bash
python train.py --mission all       # CNN + LSTM
python train.py --mission cnn       # CNN uniquement
python train.py --mission lstm      # LSTM uniquement

# Options supplémentaires
python train.py --mission cnn --epochs 50 --batch_size 64
```

### 2 · Évaluer et générer les graphiques

```bash
python evaluate.py --mission all
```

Les fichiers sont sauvegardés dans `results/` :
- `cnn_history.png` — courbes loss/accuracy
- `cnn_confusion_matrix.png` — matrice de confusion normalisée
- `lstm_history.png` — courbe MSE entraînement
- `lstm_predictions.png` — réel vs prédit + résidus

### 3 · Lancer le backend FastAPI

```bash
# Local
uvicorn main:app --reload --port 8000

# Vérifier que l'API répond
curl http://localhost:8000/health
```

Swagger UI disponible sur : **http://localhost:8000/docs**

### 4 · Ouvrir le dashboard

Double-cliquez sur `neuralbase.html` ou ouvrez-le dans votre navigateur.
Dans le panel **"Test API Live"**, l'URL est pré-remplie à `http://localhost:8000`.

---

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Statut de l'API et des modèles |
| `POST` | `/predict/cnn` | Classification image → classe CIFAR-10 |
| `POST` | `/predict/lstm` | Séquence → valeur T+1 |
| `GET` | `/docs` | Swagger UI interactif |

**Exemple — Prédiction CNN :**
```bash
curl -X POST http://localhost:8000/predict/cnn \
     -F "file=@mon_image.jpg"
```

**Exemple — Prédiction LSTM :**
```bash
curl -X POST http://localhost:8000/predict/lstm \
     -H "Content-Type: application/json" \
     -d '{"sequence": [0.12, 0.15, 0.18, 0.21, 0.24]}'
```

---

## Déploiement sur Render (gratuit)

1. Pusher le code sur GitHub
2. Créer un **Web Service** sur [render.com](https://render.com)
3. Configurer :

| Paramètre | Valeur |
|-----------|--------|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

4. Dans `neuralbase.html`, remplacer l'URL par `https://votre-app.onrender.com`

---

## Architecture des modèles

### CustomCNN (Mission 1)
- 3 blocs `Conv2D + BatchNorm + MaxPooling`
- Data Augmentation intégrée (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- Dropout (0.4) + Dense(256) → Softmax(10)
- **1 276 618 paramètres**

### LSTMForecaster (Mission 2)
- 2 couches LSTM empilées avec Dropout
- Fenêtre glissante de **24 pas** (`timeseries_dataset_from_array`)
- Normalisation `MinMaxScaler` + `inverse_transform`
- **116 033 paramètres**

---

## Exigences du projet — Checklist

- [x] Dépôt GitHub structuré (Skeleton respecté)
- [x] Modèle CNN via API Subclassing (`CustomCNN`)
- [x] Data Augmentation native (`tf.keras.layers`)
- [x] EarlyStopping + ModelCheckpoint configurés
- [x] Sliding Window (`timeseries_dataset_from_array`)
- [x] MinMaxScaler + `inverse_transform` appliqués
- [x] Matrice de confusion + courbe réelle/prédiction générées
- [x] Rapport PDF inclus
- [x] Backend FastAPI déployable sur Render

---

*Projet réalisé dans le cadre du cours de Deep Learning et réseaux de neurones — Niveau 3 SDIA*
