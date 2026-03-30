"""
models/cnn_model.py
Définition de l'architecture CNN via l'API Subclassing de Keras.
Jeu de données cible : CIFAR-10 (32x32, 10 classes)

CORRECTIF : @register_keras_serializable() + get_config() ajoutés
pour permettre le rechargement du modèle via load_model().
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="models")
class CustomCNN(Model):
    """
    CNN personnalisé pour la classification CIFAR-10.
    Architecture : 3 blocs Conv2D + MaxPooling → MLP + Dropout → Softmax
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4, **kwargs):
        super(CustomCNN, self).__init__(**kwargs)
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        # Data Augmentation (intégrée au modèle)
        self.augment = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="data_augmentation")

        # Bloc 1 : Conv2D 32 filtres
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                                   kernel_initializer="he_uniform", name="conv1")
        self.bn1   = layers.BatchNormalization(name="bn1")
        self.pool1 = layers.MaxPooling2D((2, 2), name="pool1")

        # Bloc 2 : Conv2D 64 filtres
        self.conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                                   kernel_initializer="he_uniform", name="conv2")
        self.bn2   = layers.BatchNormalization(name="bn2")
        self.pool2 = layers.MaxPooling2D((2, 2), name="pool2")

        # Bloc 3 : Conv2D 128 filtres
        self.conv3 = layers.Conv2D(128, (3, 3), padding="valid", activation="relu",
                                   kernel_initializer="he_uniform", name="conv3")
        self.bn3   = layers.BatchNormalization(name="bn3")

        # Classificateur MLP
        self.flatten      = layers.Flatten(name="flatten")
        self.dense1       = layers.Dense(256, activation="relu",
                                         kernel_initializer="he_uniform", name="dense1")
        self.dropout      = layers.Dropout(dropout_rate, name="dropout")
        self.output_layer = layers.Dense(num_classes, activation="softmax",
                                         name="output")

    def call(self, inputs, training: bool = False):
        x = self.augment(inputs, training=training)

        x = self.pool1(self.bn1(self.conv1(x), training=training))
        x = self.pool2(self.bn2(self.conv2(x), training=training))
        x = self.bn3(self.conv3(x), training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes":  self.num_classes,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_cnn(num_classes: int = 10) -> CustomCNN:
    """Instancie et retourne le modèle CustomCNN."""
    model = CustomCNN(num_classes=num_classes)
    model.build(input_shape=(None, 32, 32, 3))
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary() 
