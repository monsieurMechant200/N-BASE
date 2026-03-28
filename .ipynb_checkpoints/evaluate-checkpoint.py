"""
evaluate.py
Script d'évaluation : charge les modèles sauvegardés (.keras / .h5)
et génère tous les graphiques exigés par le rapport.

Usage :
  python evaluate.py --mission cnn
  python evaluate.py --mission lstm
  python evaluate.py --mission all
"""

import argparse
import numpy as np
import tensorflow as tf
import os

from utils.data_processing import (
    load_cifar10,
    load_jena_climate,
    create_timeseries_datasets,
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_lstm_predictions,
)

os.makedirs("results", exist_ok=True)

CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]


# ════════════════════════════════════════════════════════════════════════════
# MISSION 1 — Évaluation CNN
# ════════════════════════════════════════════════════════════════════════════

def evaluate_cnn(model_path: str = "models/saved/cnn_model.keras"):
    print("\n" + "=" * 60)
    print("  ÉVALUATION CNN (CIFAR-10)")
    print("=" * 60)

    # ── Chargement modèle ────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Lancez d'abord : python train.py --mission cnn"
        )
    model = tf.keras.models.load_model(model_path)
    print(f"[✓] Modèle chargé depuis {model_path}")

    # ── Données de test ──────────────────────────────────────────────────
    _, _, (x_test, y_test) = load_cifar10(batch_size=128)

    # ── Évaluation globale ───────────────────────────────────────────────
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Objectif 70%  : {'✅ ATTEINT' if accuracy >= 0.70 else '❌ Non atteint'}")

    # ── Prédictions → Matrice de confusion ───────────────────────────────
    print("\n[→] Génération de la matrice de confusion...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    plot_confusion_matrix(
        y_true, y_pred,
        save_path="results/cnn_confusion_matrix.png"
    )

    # ── Rapport par classe ───────────────────────────────────────────────
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES)
    print("\nRapport de classification :\n")
    print(report)

    # Sauvegarde du rapport texte
    with open("results/cnn_classification_report.txt", "w") as f:
        f.write(f"Test Accuracy : {accuracy * 100:.2f}%\n\n")
        f.write(report)
    print("[✓] Rapport sauvegardé : results/cnn_classification_report.txt")

    return accuracy


# ════════════════════════════════════════════════════════════════════════════
# MISSION 2 — Évaluation LSTM
# ════════════════════════════════════════════════════════════════════════════

def evaluate_lstm(model_path: str = "models/saved/rnn_model.keras",
                  sequence_length: int = 24):
    print("\n" + "=" * 60)
    print("  ÉVALUATION LSTM (Jena Climate)")
    print("=" * 60)

    # ── Chargement modèle ────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Lancez d'abord : python train.py --mission lstm"
        )
    model = tf.keras.models.load_model(model_path)
    print(f"[✓] Modèle chargé depuis {model_path}")

    # ── Données ──────────────────────────────────────────────────────────
    series = load_jena_climate()
    _, _, test_ds, scaler, y_test_raw, _ = create_timeseries_datasets(
        series, sequence_length=sequence_length
    )

    # ── Prédictions ──────────────────────────────────────────────────────
    print("[→] Inférence sur les données de test...")
    y_pred_norm = model.predict(test_ds, verbose=0).flatten()

    # ── Dé-normalisation (inverse_transform) ─────────────────────────────
    y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Alignement longueurs
    min_len = min(len(y_test_raw), len(y_pred))
    y_true = y_test_raw[:min_len]
    y_pred = y_pred[:min_len]

    # ── Métriques ────────────────────────────────────────────────────────
    mse  = np.mean((y_pred - y_true) ** 2)
    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)

    print(f"\n  Test MSE  : {mse:.4f}")
    print(f"  Test MAE  : {mae:.4f}")
    print(f"  Test RMSE : {rmse:.4f}")

    # ── Graphique réel vs prédit ──────────────────────────────────────────
    print("\n[→] Génération du graphique Réel vs Prédit...")
    plot_lstm_predictions(
        y_true[:500], y_pred[:500],          # Afficher les 500 premiers points
        title=f"Jena Climate — Température (MSE={mse:.4f})",
        save_path="results/lstm_predictions.png"
    )

    # Sauvegarde des métriques
    with open("results/lstm_metrics.txt", "w") as f:
        f.write(f"Test MSE  : {mse:.6f}\n")
        f.write(f"Test MAE  : {mae:.6f}\n")
        f.write(f"Test RMSE : {rmse:.6f}\n")
    print("[✓] Métriques sauvegardées : results/lstm_metrics.txt")

    return mse


# ════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation des modèles Deep Learning")
    parser.add_argument(
        "--mission",
        choices=["cnn", "lstm", "all"],
        default="all",
        help="Mission à évaluer (cnn | lstm | all)"
    )
    args = parser.parse_args()

    if args.mission in ("cnn", "all"):
        evaluate_cnn()

    if args.mission in ("lstm", "all"):
        evaluate_lstm()

    print("\n[✓] Évaluation terminée. Résultats disponibles dans le dossier results/")
