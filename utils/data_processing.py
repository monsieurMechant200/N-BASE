"""
Pipeline de données pour les deux missions :
  - Mission 1 : CIFAR-10  (tf.data.Dataset)
  - Mission 2 : Série temporelle (Sliding Window via tf.keras.utils)
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import Tuple



# MISSION 1 — CIFAR-10


def load_cifar10(batch_size: int = 64) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
    """
    Charge CIFAR-10, normalise les pixels [0,1] et renvoie des tf.data.Dataset.
    train_ds : tf.data.Dataset entraînement (batché, préfetch)
    val_ds   : tf.data.Dataset validation  (batché, préfetch)
    test_data: tuple (x_test, y_test) numpy pour l'évaluation finale
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation pixels → [0, 1] 
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Découpage train / validation (80/20)
    val_split = int(len(x_train) * 0.8)
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]

    # Construction des tf.data.Dataset
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print(f"[CIFAR-10] Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    return train_ds, val_ds, (x_test, y_test)


# MISSION 2 — Série temporelle (Jena Climate )


def load_jena_climate(csv_path: str = "data/jena_climate_2009_2016.csv",
                      column: str = "T (degC)") -> np.ndarray:
    try:
        df = pd.read_csv(csv_path)
        series = df[column].values.astype("float32")
        print(f"[Jena] Colonne '{column}' chargée : {len(series)} points.")
    except FileNotFoundError:
        print("[Jena] Fichier non trouvé — téléchargement automatique via Keras...")
        zip_path = tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname="jena_climate.csv.zip",
            extract=True,
        )
        import os
        csv_path = zip_path.replace(".zip", "")
        df = pd.read_csv(csv_path)
        series = df[column].values.astype("float32")

    return series


def create_timeseries_datasets(
    series: np.ndarray,
    sequence_length: int = 24,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset,
           MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Prépare les fenêtres glissantes pour l'entraînement LSTM.
    Étapes :
      1. Normalisation MinMaxScaler
      2. Découpage train/val/test
      3. Création tf.keras.utils.timeseries_dataset_from_array
    Retourne
    - train_ds, val_ds, test_ds : tf.data.Dataset
    - scaler                    : MinMaxScaler (pour inverse_transform)
    - y_test_raw                : valeurs réelles (non normalisées) du test
    - test_series_norm          : série normalisée (pour reconstruction graphique)
    """
    # 1. Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_norm = scaler.fit_transform(series.reshape(-1, 1))

    # 2. Découpage chronologique
    n = len(series_norm)
    n_train = int(n * train_split)
    n_val   = int(n * (train_split + val_split))

    train_series = series_norm[:n_train]
    val_series   = series_norm[n_train:n_val]
    test_series  = series_norm[n_val:]

    # Valeurs réelles pour l'évaluation visuelle
    y_test_raw = series[n_val + sequence_length:]

    # 3. Création des fenêtres glissantes
    def make_dataset(data: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
        return tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1],           # inputs
            targets=data[sequence_length:],  # target T+1
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_ds = make_dataset(train_series, shuffle=True)
    val_ds   = make_dataset(val_series,   shuffle=False)
    test_ds  = make_dataset(test_series,  shuffle=False)

    print(f"[TimeSeries] Train: {n_train} | Val: {n_val - n_train} | Test: {n - n_val} points")
    return train_ds, val_ds, test_ds, scaler, y_test_raw, test_series
