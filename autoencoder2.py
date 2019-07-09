from keras.layers import Input, Dense
from keras.models import Model

from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler


encoding_dim = 3
input_img = Input(shape=(200,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(200, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

filepath="autoencoder-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='binary_crossentropy', verbose=1, save_best_only=False)

autoencoder.summary()
autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=16,
                shuffle=True,
                validation_data=(x_train, x_train),callbacks=[checkpoint],verbose=0)

filename = "autoencoder-0.6297.hdf5"
autoencoder.load_weights(filename)
autoencoder.compile(loss='mean_squared_error', optimizer='adam')


encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)