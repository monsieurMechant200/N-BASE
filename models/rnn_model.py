import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="models")
class LSTMForecaster(Model):
    """
    Réseau LSTM pour la prédiction univariée / multivariée au pas T+1.
    """

    def __init__(self,
                 units_1: int = 128,
                 units_2: int = 64,
                 dropout_rate: float = 0.2,
                 output_size: int = 1,
                 **kwargs):
        super(LSTMForecaster, self).__init__(**kwargs)
        self.units_1      = units_1
        self.units_2      = units_2
        self.dropout_rate = dropout_rate
        self.output_size  = output_size

        self.lstm1 = layers.LSTM(
            units_1,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm1"
        )
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout1")

        self.lstm2 = layers.LSTM(
            units_2,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm2"
        )
        self.dropout2 = layers.Dropout(dropout_rate, name="dropout2")

        self.output_layer = layers.Dense(output_size, name="output")

    def call(self, inputs, training: bool = False):
        x = self.lstm1(inputs, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units_1":      self.units_1,
            "units_2":      self.units_2,
            "dropout_rate": self.dropout_rate,
            "output_size":  self.output_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_lstm(sequence_length: int = 24,
               n_features: int = 1,
               output_size: int = 1) -> LSTMForecaster:
    """Instancie et construit le modèle LSTM."""
    model = LSTMForecaster(output_size=output_size)
    model.build(input_shape=(None, sequence_length, n_features))
    return model


if __name__ == "__main__":
    model = build_lstm(sequence_length=24, n_features=1)
    model.summary()
