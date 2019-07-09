import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
import autoencoder
import tensorflow as tf


#classe qui permet de se connecter à la base de donnée
class Connect:


    db = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "root",
        port = 8889,
        database = "agenda"
    )

db = Connect.db
mycursor = db.cursor()
mycursor.execute("SELECT theme FROM theme")
result = mycursor.fetchall() #requête qui récupères toutes les données de la requête

list = []


for x in result: #méthode pour supprimer les parenthèses (ne fonctionne pas sans)

    x = str(x)
    x= x[2:]
    x = x[:-3]
    print(" \n",x)
    list.append(x)

vectorizer = CountVectorizer()
data = vectorizer.fit_transform(list) #transformation du texte en vecteur (dans toute la liste)
matrice = data.toarray()
matrice_sparse = sparse.csr_matrix(matrice) #transforme les matrices en matrice sparse
print("matrice sparse : \n",matrice_sparse, "fin matrice sparse")
longueur = (matrice_sparse[0].getnnz(1)) #récupère le nombre de ligne d'un vecteur sous matrice sparse
print(sparse.csr_matrix.todense(matrice_sparse[0]))
longueurVec = np.size(sparse.csr_matrix.todense(matrice_sparse[0])) #recupère le nombre de caractère d'un vecteur normal
print("sparse :", longueur, "\n")
print("vecteur :", longueurVec,"\n")

for i in range (len(list)):
    truc = sparse.csr_matrix.todense(matrice_sparse[i]) #retourne le vecteur sparse en normal
    print(list[i],"\n",truc,"\n maladie recupéré :",vectorizer.inverse_transform(truc)) #affiche le texte sous forme vectoriel et retourne le texte
print(" fin : \n", data.toarray()
      )
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = longueurVec
input_vec = Input(shape= np.array(matrice).shape)
print(input_vec)
encoded = Dense(encoding_dim, activation='relu')(input_vec)
print("je suis ce que je suis : ", encoded)
decoded = Dense(matrice[1],activation='sigmoid')(encoded)
autoencoder = Model(input_vec,decoded)


