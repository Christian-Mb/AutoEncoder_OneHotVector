import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
import autoencoder
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import logging
from tensorflow import set_random_seed
import os
from keras.callbacks import TensorBoard
import keras
from keras.models import load_model
import sys

# classe qui permet de se connecter à la base de donnée

class Connect:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        port=8889,
        database="mashup"
    )


class AutoEncoder:
    def __init__(self, encoding_dim=100):
        self.encoding_dim = encoding_dim
        self.x = matrice

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(15721)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='binary_crossentropy')
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        self.encoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/encoder_weights.h5')
        self.decoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/decoder_weights.h5')
        self.model.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/ae_weights.h5')

    def test(self):
        encoder = self.encoder
        decoder = self.decoder
        inputs = matrice
        x = encoder.predict(inputs)
        y = decoder.predict(x)
        print('Input: {}'.format(inputs))
        print(inputs.shape)
        print(x.shape)
        print(y.shape)
        print('Encoded: {}'.format(x))
        print('Decoded: {}'.format(y))


db = Connect.db
mycursor = db.cursor()
mycursor.execute("SELECT VUE FROM concatenantion ")
result = mycursor.fetchall()  # requête qui récupères toutes les données de la requête

list = []

for x in result:  # méthode pour supprimer les parenthèses (ne fonctionne pas sans)
    x = str(x)
    x = x[2:]
    x = x[:-3]
    list.append(x)

vectorizer = CountVectorizer()
data = vectorizer.fit_transform(list)  # transformation du texte en vecteur (dans toute la liste)

matrice = data.toarray()
matrice_sparse = sparse.csr_matrix(matrice)  # transforme les matrices en matrice sparse
longueur = (matrice_sparse[0].getnnz(1))  # récupère le nombre de ligne d'un vecteur sous matrice sparse
longueurVec = np.size(sparse.csr_matrix.todense(matrice_sparse[0]))  # recupère le nombre de caractère d'un vecteur normal


ae = AutoEncoder(encoding_dim=100)
ae.encoder_decoder()
ae.fit(batch_size=300, epochs=100)
ae.save()
ae.test()

encoder = load_model(r'C:/Users/Christian Mada-Mbari/Desktop/test/encoder_weights.h5')
decoder = load_model(r'C:/Users/Christian Mada-Mbari/Desktop/test/decoder_weights.h5')


x = encoder.predict(matrice)
y = decoder.predict(x)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
#print(matrice)
print("y a la base: \n",y)
print(y > 0.5)
y[(y>0.5)] = 1
y[(y<0.5)] = 0
print("y après : \n",y)
print(vectorizer.inverse_transform(y[0]))
# for i in range (6375):
#             for j in range(16217):
#                 if matrice[i,j] == 0:
#                     if abs(matrice[i,j] - y[i,j] < 0.5):
#                         y[i,j] = 0
#                     else:
#                         y[i,j] = 1
#                 if matrice[i,j] == 1:
#                     if abs(matrice[i,j] - y[i,j] > 0.5):
#                         y[i,j] = 1
#                     else:
#                         y[i,j] = 0

#print('Input: {}'.format(matrice))
#print('Encoded: {}'.format(x))
#print('Decoded: {}'.format(y))
#print("format de 0: \n",format(y[0]))
#print("format de 1: \n",format(y[1]))
#print(vectorizer.inverse_transform(format(y)))
#print(vectorizer.inverse_transform(format(y[1])))
print(y.shape)
print(vectorizer.inverse_transform(y[50]))