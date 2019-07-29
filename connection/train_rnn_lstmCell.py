import time

import keras
from lib.LSTM_data_preprocessing import load_data, prepare_data
from lib.utils import calculate_accuracy

from deeplearning_models import rnn_lstm

if __name__ == '__main__':
    global_start_time = time.time()
    Epochos = 1000
    seq_len = 1
    service_size = 10

    print('> prepare data ...')

    data_file = './datasets/final_Mashup3.txt'
    prepare_data(data_file)

    filename = './datasets/LSTM_data_final2.csv'

    print('> load data ...')

    X_train, Y_train, X_test, Y_test = load_data(filename,seq_len,service_size=service_size)

    print('> Data Loaded. Compiling...')

    model = rnn_lstm.build_model([service_size, seq_len, 2*seq_len, service_size])

    #Tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorboard_summary/LSTM/5', histogram_freq=10, write_graph=True, write_images=True)

    model.fit(X_train, Y_train, batch_size=10, nb_epoch=Epochos, validation_split=0.05,callbacks=[tbCallBack])

    Y_predicted = rnn_lstm.predict_point_by_point(model,X_test)

    print(calculate_accuracy(Y_test,Y_predicted))
    for i in range(10):
        print('Desire output',Y_test[i])
        print('Real output',Y_predicted[i])
