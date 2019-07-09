import keras
from keras.layers import Input, Dense
from keras.models import Model
import os
import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse



class Connect:


    db = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "root",
        port = 8889,
        database = "agenda"
    )


class AutoEncoder:

    def oneHot(self):

        db = Connect.db
        mycursor = db.cursor()
        mycursor.execute("SELECT theme FROM theme")
        result = mycursor.fetchall()  # requête qui récupères toutes les données de la requête

        list = []

        for x in result:  # méthode pour supprimer les parenthèses (ne fonctionne pas sans)

            x = str(x)
            x = x[2:]
            x = x[:-3]
            print(" \n", x)
            list.append(x)

        vectorizer = CountVectorizer()
        data = vectorizer.fit_transform(list)  # transformation du texte en vecteur (dans toute la liste)
        matrice = data.toarray()
        matrice_sparse = sparse.csr_matrix(matrice)  # transforme les matrices en matrice sparse
        return matrice





    def __init__(self, encoding_dim=3):

        self.encoding_dim = encoding_dim
        self.x = self.oneHot()
        print(self.x)



    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(39)(inputs)
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
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'C:/Users/Christian Mada-Mbari/Desktop/test/'):
            os.mkdir(r'C:/Users/Christian Mada-Mbari/Desktop/test/')
        else:
            self.encoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/encoder_weights.h5')
            self.decoder.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/decoder_weights.h5')
            self.model.save(r'C:/Users/Christian Mada-Mbari/Desktop/test/ae_weights.h5')

    def test(self):
        encoder = self.encoder
        decoder = self.decoder

        input = self.oneHot()
        x = encoder.predict(input)
        y = decoder.predict(x)

        print("Input: {} \n".format(input))
        print("\nEncoded: {} \n".format(x))
        print("\nDecoded: {} \n".format(y))

if __name__ == '__main__':

    ae = AutoEncoder(encoding_dim=39)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=300)
    ae.save()
    ae.test()

