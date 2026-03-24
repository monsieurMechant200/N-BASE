#!/usr/bin/env python
# coding: utf-8

# In[20]:


from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# # visualisation des donnees

# In[12]:


# Chargement du jeu de données
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Définition des noms des classes 
class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']


# 3. Affichage des donnes
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])# 0  represnte le premier element
plt.show()

# Vérification de la forme des données
print(f"x_train forme : {x_train.shape}, y_train shape : {y_train.shape} ")
print(f"x_test : {x_test.shape}, y_test shape : {y_test.shape} ")





# # Prétraitement des données

# In[19]:


# Normalisation des valeurs des pixels à la plage [0, 1] en divisant par 255
#On utilise 255.0 (avec un point) pour s'assurer que Python transforme les nombres en nombres à virgule (float) et non en entiers
x_train=  x_train / 255.0
x_test = x_test / 255.0

# applair les donnees
#.flatten() est une méthode de la bibliothèque NumPy qui sert à transformer un tableau multidimensionnel (matrice) en un tableau à une seule dimension (vecteur)
y_train = y_train.flatten()
y_test = y_test.flatten()

print('y_train est :',y_train)
print('y_test est :',y_test)
print('x_train est :',x_train)
print('x_test est :',x_test)


# # implémentation des couches de data augmentation (RandomFlip, RandomRotation)

# In[23]:


# Définir les couches d'augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
])

# construction du pipeline tf.data.Dataset.
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(500).batch(20)             #shuffle(500)signifie qu'On prend les 1000 premières images, on les mélange bien, et on en pioche au hasard
                                                       # batch(20)represente le regroupement en paquet de  20 . le modele va regarer 20 image au meme moment
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds 


# In[ ]:




