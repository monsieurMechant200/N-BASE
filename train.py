"""
train.py
Script principal d'entraînement pour les deux missions.

Usage :
  python train.py --mission cnn
  python train.py --mission lstm
  python train.py --mission all
"""

import argparse
import os
import tensorflow as tf

# Imports internes
from models.cnn_model  import build_cnn
from models.rnn_model  import build_lstm
from utils.data_processing import (
    load_cifar10,
    load_jena_climate,
    create_timeseries_datasets,
)
from utils.visualization import (
    plot_training_history,
    plot_lstm_history,
)

os.makedirs("models/saved", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Callbacks communs


def get_callbacks(monitor: str = "val_loss",
                  model_path: str = "models/saved/best_model.h5"):
    """
    Retourne une liste de callbacks standards :
      - EarlyStopping  : arrêt si pas d'amélioration pendant 10 époques
      - ModelCheckpoint: sauvegarde le meilleur modèle
      - ReduceLROnPlateau: réduit le LR si plateau
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# MISSION 1 — Entraînement CNN


def train_cnn(epochs: int = 20, batch_size: int = 64):
    print("  MISSION 1 — Entraînement CNN (CIFAR-10)")


    # Données
    train_ds, val_ds, _ = load_cifar10(batch_size=batch_size)

    # Modèle
    model = build_cnn(num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    # Entraînement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(
            monitor="val_loss",
            model_path="models/saved/cnn_model.h5"
        ),
        verbose=1,
    )

    # Sauvegarde + Visualisation
    model.save("models/saved/cnn_model.keras")
    plot_training_history(history, save_path="results/cnn_history.png")
    print("\n[✓] CNN entraîné et sauvegardé dans models/saved/cnn_model.keras")
    return history



# MISSION 2 — Entraînement LSTM



def train_lstm(epochs: int = 20,
               batch_size: int = 32,
               sequence_length: int = 24):
    print("  MISSION 2 — Entraînement LSTM (Jena Climate)")

    # Chargement des Données
    series = load_jena_climate()
    train_ds, val_ds, test_ds, scaler, _, _ = create_timeseries_datasets(
        series,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    # determinons les n_features dynamiquement à partir d'un echantillon de dataset
    for inputs, targets in train_ds.take(1):
        input_shape = inputs.shape
        n_features = input_shape[-1]
        print(f"DEBUG: Input shape détectée: {input_shape}")

    # Modèle
    model = build_lstm(sequence_length=sequence_length, n_features=n_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    # Entraînement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(
            monitor="val_loss",
            model_path="models/saved/rnn_model.h5"
        ),
        verbose=1,
    )

    # Sauvegarde + Visualisation
    model.save("models/saved/rnn_model.keras")
    plot_lstm_history(history, save_path="results/lstm_history.png")
    print("\n LSTM entraîné et sauvegardé dans models/saved/rnn_model.h5")
    return history


# Point d'entrée

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Deep Learning")
    parser.add_argument(
        "--mission",
        choices=["cnn", "lstm", "all"],
        default="all",
        help="Mission à entraîner (cnn | lstm | all)"
    )
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_known_args()[0]

    if args.mission in ("cnn", "all"):
        train_cnn(epochs=args.epochs, batch_size=args.batch_size)

    if args.mission in ("lstm", "all"):
        train_lstm(epochs=args.epochs, batch_size=args.batch_size)
