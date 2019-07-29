import os
import sys
import keras
import mysql.connector
import numpy
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

'''Cette classe permet de connecter à la base de données
    Avant toute utilisation verifier que la vous aviez le bon, user, port , passwd, et database
'''

class Connect:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        port=8889,
        database="mashup"
    )

'''Cette classe permet de recupérer les données et de les transformer en matrice grâce à One hot
'''
class OneHot :

    def __int__(self):
        self.x_train = self.matrice

    '''
    Cette méthode retoune la matrice codée à chaud et son vectorizer'''
    def One_Hot(self):
        db = Connect.db
        mycursor = db.cursor()
        #si autre de de données, modifier la requête en conséquent
        mycursor.execute("SELECT VUE FROM concatenantion ")
        result = mycursor.fetchall()  # requête qui récupères toutes les données de la requête

        list = []
        for x in result:  # méthode pour supprimer les parenthèses (ne fonctionne pas sans)
            x = str(x)
            x = x[2:]
            x = x[:-3]
            list.append(x)
        vectorizer = CountVectorizer()


        # transformation du texte en vecteur (dans toute la liste)
        data = vectorizer.fit_transform(list)
        self.matrice = data.toarray()
        print(self.matrice)
        self.vector = vectorizer
        return self.matrice, self.vector

'''Cette classe permet de compresser et décompresser la matrice codée à chaud'''
class AutoEncoder :
    def __init__(self, encoding_dim=100):
        self.encoding_dim = encoding_dim
        #recupération de la matrice
        ae = OneHot()
        self.x , self.vector = ae.One_Hot()



    '''Méthode pour compresser la matrice et retoune la matrice compressée'''
    def _encoder(self):
        #on récupère le nombre de ligne de la matrice
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    '''Méthode pour décompresser la matrice et retoune la matrice décompressée'''
    def _decoder(self):
        # on récupère le nombre de ligne de la matrice
        inputs = Input(shape=(self.encoding_dim,))
        #on recupère le nombre de colonnes de la matrice
        decoded = Dense(self.x[0].shape[0], activation= 'sigmoid')(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model
    "Appel des deux méthodes précédentes"
    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    '''Permet de lancer l'entrainement'''
    def fit(self, batch_size=10, epochs=100):
        self.model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack]),



    '''Permet de sauvegarder mes resultats de l'entrainemen mais il faut modifier la direction du fichier'''
    def save(self):
        if not os.path.exists(r'C:/Users/Christian Mada-Mbari/Desktop/test/'):
            os.mkdir(r'C:/Users/Christian Mada-Mbari/Desktop/test/')
        else:
            self.encoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/encoder_weights.h5')
            self.decoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/decoder_weights.h5')
            self.model.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/ae_weights.h5')


    '''Permet de lancer le programme avec l'entainement'''
    def runWithTraining(self):
        self.encoder_decoder()
        self.fit(batch_size=10, epochs=100)
        self.save()
        np.set_printoptions(suppress= True)
        #print( self.vector.build_tokenizer())
        encoder = self.encoder
        decoder = self.decoder
        inputs = self.x


        x = encoder.predict(inputs)
        y = decoder.predict(x)

        #permet de compenser les pertes lors de la décompression
        y[(y >= 0.5 )] = 1
        y[(y < 0.5)] = 0

        numpy.set_printoptions(threshold=sys.maxsize)
        print("debut", self.vector.inverse_transform(self.x[0]))
        print(self.vector.inverse_transform(y[0]))


        #print(inputs.shape)
       # print(x.shape)
       # print(y.shape)
        #print('Encoded: {}'.format(x))
        #print('Decoded: {}'.format(y))

    '''Permet de lancer le programme sans l'entainement'''
    def runWithoutTraining(self):
        encoder = load_model(r'C:/Users/Christian Mada-Mbari/Desktop/test/encoder_weights.h5')
        decoder = load_model(r'C:/Users/Christian Mada-Mbari/Desktop/test/decoder_weights.h5')

        x =  encoder.predict(self.x)
        y =  decoder.predict(x)

        # permet de compenser les pertes lors de la décompression
        y[(y >= 0.5)] = 1
        y[(y < 0.5)] = 0

        numpy.set_printoptions(threshold=sys.maxsize)
        print("Avant la transformation : ",self.vector.inverse_transform(self.x[0]))
        print("Après la transformation : ",self.vector.inverse_transform(y[0]))



if __name__ == '__main__':

    ae = AutoEncoder(encoding_dim=100)
    '''Commenter où décommenter l'une des lignes suivantes pour faire le test avec ou sans l'entrainement'''
    ae.runWithTraining()
    #ae.runWithoutTraining()

