"""
utils/visualization.py
Fonctions de visualisation pour les deux missions :
  - Courbes de perte (Training History)
  - Matrice de confusion (CNN)
  - Courbe réelle vs prédiction (LSTM)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import itertools
import os

# Palette commune
COLORS = {
    "train":     "#6366f1",   # Indigo
    "val":       "#10b981",   # Emerald
    "actual":    "#94a3b8",   # Slate
    "predicted": "#f59e0b",   # Amber
    "bg":        "#0f172a",
    "card":      "#1e293b",
    "text":      "#f8fafc",
    "muted":     "#94a3b8",
    "grid":      "#334155",
}

CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["card"],
    "axes.edgecolor":    COLORS["grid"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["muted"],
    "ytick.color":       COLORS["muted"],
    "text.color":        COLORS["text"],
    "grid.color":        COLORS["grid"],
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
})


def save_fig(fig: plt.Figure, path: str):
    """Sauvegarde la figure et affiche le chemin."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[✓] Figure sauvegardée : {path}")
    plt.close(fig)



# MISSION 1 — CNN


def plot_training_history(history, save_path: str = "results/cnn_history.png"):
    """
    Trace les courbes Train Loss / Val Loss et Train Acc / Val Acc.

    Paramètres
    ----------
    history : objet retourné par model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN — Historique d'entraînement", fontsize=14, color=COLORS["text"])

    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history.history["loss"], color=COLORS["train"],
            linewidth=2, label="Train Loss")
    ax.plot(epochs, history.history["val_loss"], color=COLORS["val"],
            linewidth=2, linestyle="--", label="Val Loss")
    ax.set_title("Perte (Loss)", color=COLORS["text"])
    ax.set_xlabel("Époque")
    ax.set_ylabel("Sparse Categorical Crossentropy")
    ax.legend(framealpha=0.3, facecolor=COLORS["card"])
    ax.grid(True, alpha=0.4)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history.history["accuracy"], color=COLORS["train"],
            linewidth=2, label="Train Acc")
    ax.plot(epochs, history.history["val_accuracy"], color=COLORS["val"],
            linewidth=2, linestyle="--", label="Val Acc")
    ax.axhline(y=0.70, color="#f43f5e", linestyle=":", linewidth=1.5,
               label="Objectif 70%")
    ax.set_title("Précision (Accuracy)", color=COLORS["text"])
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.3, facecolor=COLORS["card"])
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_fig(fig, save_path)


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str = "results/cnn_confusion_matrix.png"):
    """
    Trace et sauvegarde la matrice de confusion normalisée pour CIFAR-10.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_title("Matrice de Confusion — CNN (CIFAR-10)", fontsize=14)
    ax.set_xlabel("Classe Prédite")
    ax.set_ylabel("Classe Réelle")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else COLORS["muted"],
                fontsize=8)

    plt.tight_layout()
    save_fig(fig, save_path)



# MISSION 2 — LSTM


def plot_lstm_predictions(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          title: str = "Jena Climate — Température",
                          save_path: str = "results/lstm_predictions.png"):
    """
    Superpose la courbe réelle et la courbe prédite (valeurs dé-normalisées).

    Paramètres
    ----------
    y_true : valeurs réelles après inverse_transform
    y_pred : valeurs prédites après inverse_transform
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"LSTM — {title}", fontsize=14, color=COLORS["text"])

    x = np.arange(len(y_true))

    # Courbes superposées
    ax = axes[0]
    ax.plot(x, y_true, color=COLORS["actual"],   linewidth=1.5,
            alpha=0.8, label="Valeur Réelle")
    ax.plot(x, y_pred, color=COLORS["predicted"], linewidth=1.5,
            alpha=0.9, label="Prédiction T+1")
    ax.fill_between(x, y_true, y_pred, alpha=0.12, color=COLORS["predicted"])
    ax.set_title("Prédiction vs Réalité", color=COLORS["text"])
    ax.set_ylabel("Valeur (unité originale)")
    ax.legend(framealpha=0.3, facecolor=COLORS["card"])
    ax.grid(True, alpha=0.4)

    # Résidus
    ax = axes[1]
    residuals = y_pred - y_true
    ax.fill_between(x, residuals, alpha=0.5, color=COLORS["val"])
    ax.axhline(0, color=COLORS["text"], linewidth=0.8, linestyle="--")
    ax.set_title("Résidus (Prédiction − Réalité)", color=COLORS["text"])
    ax.set_xlabel("Pas de temps (test)")
    ax.set_ylabel("Erreur")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_fig(fig, save_path)


def plot_lstm_history(history, save_path: str = "results/lstm_history.png"):
    """Trace les courbes MSE Train / Val pour le LSTM."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history.history["loss"]) + 1)

    ax.plot(epochs, history.history["loss"],     color=COLORS["train"],
            linewidth=2, label="Train MSE")
    ax.plot(epochs, history.history["val_loss"], color=COLORS["val"],
            linewidth=2, linestyle="--", label="Val MSE")
    ax.set_title("LSTM — Historique d'entraînement (MSE)", fontsize=13)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Mean Squared Error")
    ax.legend(framealpha=0.3, facecolor=COLORS["card"])
    ax.grid(True, alpha=0.4)
    fig.suptitle("")
    plt.tight_layout()
    save_fig(fig, save_path)
