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

# Import des classes custom 
# pour désérialiser correctement le fichier .keras.
# Le décorateur @register_keras_serializable dans chaque fichier modèle
try:
    from models.cnn_model import CustomCNN, ResBlock
except ImportError:
    from models.cnn_model import CustomCNN
    ResBlock = None   
from models.rnn_model import LSTMForecaster         

CUSTOM_OBJECTS = {
    "CustomCNN":      CustomCNN,
    "ResBlock":       ResBlock,
    "LSTMForecaster": LSTMForecaster,
}

os.makedirs("results", exist_ok=True)

CIFAR10_CLASSES = [
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]



def resolve_model_path(primary: str) -> str:
    if os.path.exists(primary):
        return primary

    if primary.endswith(".keras"):
        fallback = primary[:-6] + ".h5"
    elif primary.endswith(".h5"):
        fallback = primary[:-3] + ".keras"
    else:
        fallback = None

    if fallback and os.path.exists(fallback):
        print(f"  [!] '{primary}' introuvable → fallback : '{fallback}'")
        return fallback

    raise FileNotFoundError(
        f"Modèle introuvable : {primary}"
        + (f"  (fallback '{fallback}' aussi absent)" if fallback else "")
        + "\nLancez d'abord : python train.py --mission cnn|lstm"
    )

#  chargement robuste avec custom_objects


def load_model_safe(model_path: str, label: str) -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
        print(f" {label} chargé depuis : {model_path}")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger le modèle {label} depuis '{model_path}'.\n"
            f"Détail : {exc}\n\n"
            "Causes fréquentes :\n"
            "   Le .keras a été entraîné avec une architecture différente\n"
            "    de celle définie dans models/cnn_model.py (shape mismatch).\n"
            "    Ré-entraînez avec : python train.py --mission cnn\n"
            "   Les classes custom (ResBlock, CustomCNN…) ne correspondent pas.\n"
            "    Vérifiez que models/cnn_model.py n'a pas été modifié depuis l'entraînement."
        ) from exc

# Évaluation CNN


def evaluate_cnn(model_path: str = "models/saved/cnn_model.keras"):
    print("  ÉVALUATION CNN (CIFAR-10)")

    model_path = resolve_model_path(model_path)
    model      = load_model_safe(model_path, "CNN")

    # Données de test
    _, _, (x_test, y_test) = load_cifar10(batch_size=128)

    # Recompilation (nécessaire après compile=False)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(f"  Objectif 70%  : {'ATTEINT' if accuracy >= 0.70 else 'Non atteint'}")

    # Prédictions → Matrice de confusion
    print("\n[→] Génération de la matrice de confusion...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    plot_confusion_matrix(
        y_true, y_pred,
        save_path="results/cnn_confusion_matrix.png"
    )

    # Rapport par classe
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES)
    print("\nRapport de classification :\n")
    print(report)

    with open("results/cnn_classification_report.txt", "w") as f:
        f.write(f"Modèle chargé   : {model_path}\n")
        f.write(f"Test Accuracy   : {accuracy * 100:.2f}%\n")
        f.write(f"Test Loss       : {loss:.4f}\n\n")
        f.write(report)
    print("Rapport sauvegardé : results/cnn_classification_report.txt")

    return accuracy

# MISSION 2 — Évaluation LSTM

def evaluate_lstm(model_path: str = "models/saved/rnn_model.keras",
                  sequence_length: int = 24):
    print("=" * 60)
    print("  ÉVALUATION LSTM (Jena Climate)")
    print("=" * 60)

    model_path = resolve_model_path(model_path)
    model      = load_model_safe(model_path, "LSTM")
    series = load_jena_climate()
    _, _, test_ds, scaler, y_test_raw, _ = create_timeseries_datasets(
        series, sequence_length=sequence_length
    )
    print("[→] Inférence sur les données de test...")
    y_pred_norm = model.predict(test_ds, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    min_len = min(len(y_test_raw), len(y_pred))
    y_true  = y_test_raw[:min_len]
    y_pred  = y_pred[:min_len]
    mse  = np.mean((y_pred - y_true) ** 2)
    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)

    print(f"\n  Test MSE  : {mse:.4f}")
    print(f"  Test MAE  : {mae:.4f}")
    print(f"  Test RMSE : {rmse:.4f}")
    print("\n Génération du graphique Réel vs Prédit...")
    plot_lstm_predictions(
        y_true[:500], y_pred[:500],
        title=f"Jena Climate — Température (MSE={mse:.4f})",
        save_path="results/lstm_predictions.png"
    )

    #Sauvegarde des métriques 
    with open("results/lstm_metrics.txt", "w") as f:
        f.write(f"Modèle chargé : {model_path}\n\n")
        f.write(f"Test MSE  : {mse:.6f}\n")
        f.write(f"Test MAE  : {mae:.6f}\n")
        f.write(f"Test RMSE : {rmse:.6f}\n")
    print("[✓] Métriques sauvegardées : results/lstm_metrics.txt")

    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation des modèles Deep Learning")
    parser.add_argument(
        "--mission",
        choices=["cnn", "lstm", "all"],
        default="all",
        help="Mission à évaluer (cnn | lstm | all)"
    )
    parser.add_argument(
        "--cnn_path",
        type=str,
        default="models/saved/cnn_model.keras",
        help="Chemin vers le modèle CNN (.keras ou .h5)"
    )
    parser.add_argument(
        "--lstm_path",
        type=str,
        default="models/saved/rnn_model.keras",
        help="Chemin vers le modèle LSTM/RNN (.keras ou .h5)"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=24,
        help="Longueur de séquence LSTM utilisée à l'entraînement (défaut : 24)"
    )
    # parse_known_args() ignore les arguments inconnus (ex: -f kernel-xxx.json de Jupyter)
    args = parser.parse_known_args()[0]

    if args.mission in ("cnn", "all"):
        evaluate_cnn(model_path=args.cnn_path)

    if args.mission in ("lstm", "all"):
        evaluate_lstm(
            model_path=args.lstm_path,
            sequence_length=args.sequence_length,
        )

    print("\n Évaluation terminée. Résultats disponibles dans le dossier results/") 
