import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
class LSTMDataPreparer:
    def __init__(self, sequence_length: int = 50, test_size: float = 0.2): 
        #  sequence_length: longueur de la fenêtre temporelle
        if sequence_length <= 0:
            raise ValueError("sequence_length doit être > 0") # le programme déclenche volontairement un message d'erreur
        if not (0 < test_size < 1):
            raise ValueError("test_size doit être entre 0 et 1") # l'erreur est enlevé si test_size est hors de l'intervalle [0,1]
        self.sequence_length = sequence_length # Stocke dans l’attribut sequence_length de l’objet la valeur passée en paramètre au constructeur 
        self.test_size = test_size # Stocke dans l’attribut test_size la proportion ou le nombre d’échantillons réservés pour le jeu de test
        self.scaler = MinMaxScaler(feature_range=(0, 1)) #  Crée un objet MinMaxScaler de scikit-learn et le stocke dans l’attribut scaler.

    def normalize(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray): # vérifie si data n'est pas une instance de numpy
            raise TypeError("Les données doivent être un numpy.ndarray") 
        return self.scaler.fit_transform(data) # retourne les données transformées

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: # retourne un tuple de 2tableaux numpy

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def split_train_test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
 
        split_index = int(len(X) * (1 - self.test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def prepare(self, raw_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        normalized_data = self.normalize(raw_data)
        X, y = self.create_sequences(normalized_data)
        return self.split_train_test(X, y)





