#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[2]:


url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

fname = keras.utils.get_file("jena_climate_2009_2016.csv.zip", origin=url, extract=True)
# keras.utils.get_file: est une fonction de keras permettant de télécharger automatiquement un fichier csv et le sauvegarder
# extract = True: signifie que si le fichier est compressé, on doit le décompresser
print(fname)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[6]:


df = pd.read_csv(r"C:\Users\NT Store\.keras\datasets\jena_climate_2009_2016.csv.zip")
df.head(12)
df.to_csv("Documents\ jena_climate_2009_2016.csv", index=False ) 


# In[ ]:





# In[7]:


from sklearn.preprocessing import MinMaxScaler


# In[8]:


# On garde 1 mesure par heure (les données sont toutes les 10 min)
# iloc[::6] = on prend une ligne sur 6
temperature = df["T (degC)"].values[::6]
# df["T (degC)"]: sélection la colonne T (degC)
# .values converti en un tableau numpy plus léger
# [::6]: sélectionne une valeur sur 6 car[début:fin:pas] les données étant en 10min, on aura 1 mesure par heure (6*10 min)
print(f"Nombre d'heures : {len(temperature)}")  # ~70 000 heures
print(f"Température min : {temperature.min():.1f}°C") # .f est un f-string qui indique le nbre de chiffre après la virguel
print(f"Température max : {temperature.max():.1f}°C")


# In[9]:



# Visualisation 
plt.figure(figsize=(12, 3)) # crée une figure eg définit la taille en pouce ( largeur * hauteur)
plt.plot(temperature[:24*30])   # trace la courbe des températures (30 premiers jours ie 720 premières observations). l'axe x est automatiquement l'index(0...719)
plt.title("Température — 30 premiers jours")
plt.xlabel("Heure")
plt.ylabel("°C")
plt.tight_layout() # permet d'éviter le chevauchement des titres des labels
plt.show()


# In[10]:


# ÉTAPE 3 : SÉPARATION TRAIN / TEST
# On utilise 80% pour l'entraînement, 20% pour le test
n = len(temperature)
n_train = int(n * 0.80)

train_data = temperature[:n_train]       # les 80% premiers (passé)
test_data  = temperature[n_train:]       # les 20% derniers (futur)

print(f"\nTrain : {len(train_data)} heures")
print(f"Test  : {len(test_data)} heures")


# In[11]:


# ÉTAPE 4 : NORMALISATION avec MinMaxScaler

# MinMaxScaler ramène toutes les valeurs entre 0 et 1
# Formule : x_norm = (x - min) / (max - min)
scaler = MinMaxScaler() # crée un objet vide MinMaxScaler et le stocke sur la variable scaler

scaler.fit(train_data.reshape(-1, 1))   # reshape car MinMaxScaler attend 2D, -1 permet de calculer automatiquement le nombre de lignes
# .fit permet de calculer le min et le mx de chaque colonne uniquement utilisé dans le train


# In[12]:


# On applique ensuite la même normalisation sur train ET test
train_scaled = scaler.transform(train_data.reshape(-1, 1)).flatten() #
# .transfor(): oblige un tableau 2D, reshape permet d'avoir un tableau 2D
# scaler.transform(train_data.reshape(-1, 1)): applique la formule de normalisation
# .flatten() reconverti le tableau 2D en 1D
test_scaled  = scaler.transform(test_data.reshape(-1, 1)).flatten()

print(f"\nAprès normalisation :")
print(f"Train — min : {train_scaled.min():.3f}  max : {train_scaled.max():.3f}")
print(f"Test  — min : {test_scaled.min():.3f}   max : {test_scaled.max():.3f}")


# In[13]:


# ÉTAPE 5 : FENÊTRES GLISSANTES avec timeseries_dataset_from_array

# Principe : pour prédire l'heure t+1,
# on donne au modèle les WINDOW_SIZE heures précédentes
WINDOW_SIZE = 24    # le modèle garde les 24 heures passées pour prédire l'heure suivante : c'est la taille de la fenêtre de contexte
BATCH_SIZE  = 32    # au lieu de traiter les exemples un par un, le modèle en traite 32 à la fois. Plus rapide et plus stable pour l'apprentissage

# --- Dataset d'entraînement ---
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data     = train_scaled[:-1],   # entrées : toutes les heures sauf la dernière
    targets  = train_scaled[1:],    # cibles  : décalées d'1 heure vers l'avenir
    sequence_length = WINDOW_SIZE,  # taille de chaque fenêtre
    batch_size      = BATCH_SIZE,   # nbre de fenêtre regroupé ensenble 
    shuffle         = True,         # mélanger les fenetres à chaque époque pour l'entraînement
    seed            = 42,           # permet de fixer le hasard de tellesorte qu'il soit reproductif
)

# --- Dataset de test ---
test_ds = tf.keras.utils.timeseries_dataset_from_array(
    data     = test_scaled[:-1],
    targets  = test_scaled[1:],
    sequence_length = WINDOW_SIZE,
    batch_size      = BATCH_SIZE,
    shuffle         = False,        # ne pas mélanger pour le test
)


# Vérification des dimensions
for X, y in train_ds.take(1):
    print(f"\nForme d'un batch X : {X.shape}")  # (32, 24, 1) → 32 fenêtres de 24h
    print(f"Forme d'un batch y : {y.shape}")    # (32,)       → 32 valeurs cibles


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:





# In[ ]:





# In[ ]:





# In[ ]:




