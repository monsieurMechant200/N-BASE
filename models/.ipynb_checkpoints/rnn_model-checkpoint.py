"""
models/rnn_model.py
Définition de l'architecture LSTM pour la prédiction de série temporelle (T+1).
Compatible avec tout dataset normalisé via MinMaxScaler.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class LSTMForecaster(Model):
    """
    Réseau LSTM pour la prédiction univariée / multivariée au pas T+1.

    Paramètres
    ----------
    units_1 : int   — Nombre d'unités dans la 1ʳᵉ couche LSTM
    units_2 : int   — Nombre d'unités dans la 2ᵉ couche LSTM
    dropout_rate : float — Taux de dropout entre les couches
    output_size  : int  — Nombre de valeurs à prédire (1 pour univarié)
    """

    def __init__(self,
                 units_1: int = 128,
                 units_2: int = 64,
                 dropout_rate: float = 0.2,
                 output_size: int = 1):
        super(LSTMForecaster, self).__init__()

        # Couche 1 LSTM : return_sequences=True → transmet la séquence
        self.lstm1 = layers.LSTM(
            units_1,
            return_sequences=True,   # Nécessaire pour empiler une 2ᵉ couche
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm1"
        )
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout1")

        # Couche 2 LSTM : return_sequences=False → ne garde que l'état final
        self.lstm2 = layers.LSTM(
            units_2,
            return_sequences=False,  # ⚠ Exigé par le sujet pour prédire T+1
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm2"
        )
        self.dropout2 = layers.Dropout(dropout_rate, name="dropout2")

        # Couche Dense finale : prédiction T+1
        self.output_layer = layers.Dense(output_size, name="output")

    def call(self, inputs, training: bool = False):
        x = self.lstm1(inputs, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


def build_lstm(sequence_length: int = 24,
               n_features: int = 1,
               output_size: int = 1) -> LSTMForecaster:
    """
    Instancie et construit le modèle LSTM.

    Paramètres
    ----------
    sequence_length : longueur de la fenêtre glissante
    n_features      : nombre de variables (1 = univarié)
    output_size     : nombre de valeurs à prédire
    """
    model = LSTMForecaster(output_size=output_size)
    model.build(input_shape=(None, sequence_length, n_features))
    return model


# Test rapide
if __name__ == "__main__":
    model = build_lstm(sequence_length=24, n_features=1)
    model.summary()
