import tensorflow as tf
from lib.API_data_preprocessing import prepare_dataset
from lib.utils import generate_batches, calculate_accuracy
from math import ceil

from deeplearning_models.denoising_autoencoder import DenoisingAutoencoder

network_architecture = dict(layers=[], n_input=int())
dataset_name = 'final_API_data.csv'
dataset, X_train, X_test = prepare_dataset('./datasets/' + dataset_name)
import pandas as pd
dataset = pd.DataFrame(X_train[:2500])
dataset.to_csv('./datasets/dataset_after_datapreprocess.csv')

API_Names = list(dataset.iloc[:, 0].values)
dataset = dataset.iloc[:, 1:]

teX = X_test.iloc[:, 1:]
trX = X_train.iloc[:, 1:]

corrupt_rate = [1.0]

def prepare_mashup(data):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    words = set()
    API_names = list(data.keys())
    transormed_API_names = le.fit_transform(API_names)

    #dict_names = {k:v for k,v in zip(API_names,transormed_API_names)}


    with open('./datasets/'+'API_mashup.txt','r') as f:
        mashup_file = open("./datasets/"+'final_Mashup4.txt','w')

        for line in f.readlines():
            names = line.split('\\')
            newLine = ''
            for i, name in enumerate(names):
                if i > 0:    newLine += '||'
                name = name.lstrip().rstrip()

                if name in data:
                    #newLine += str(dict_names[name])+','
                    for value in data[name].tolist():
                        newLine += str(value)+','
                else:
                    words.add(name)
                    newLine += name

            mashup_file.write(newLine+'\n')

        mashup_file.close()
        print('number of API names in our Datast',len(data))
        print(" number of API names in Mashup",len(words))

        dataset_after_DR = []
        for key, value in data.items():
            value = value.tolist()
            value.append(key)
            dataset_after_DR.append(value)
        dataset_after_DR = pd.DataFrame(dataset_after_DR)
        dataset_after_DR.to_csv('./datasets/dataset_after_DR.csv')


def save_reduced_dimensions(data_dict={}):
    from lib.__init_db__ import execute_query

    QUERY = 'TRUNCATE `api_service_table`'
    response = execute_query(QUERY)
    print(len(data_dict.keys()))

    for key,value in data_dict.items():
        #print(key)
        QUERY = 'INSERT INTO `api_service_table` (`API_NAME`, `REDUCED_VERSION`) VALUES (\'%API_NAME%\', \'%VALUE%\');'
        QUERY = QUERY.replace('%API_NAME%',key).replace('%VALUE%',str(value))
        data = execute_query(QUERY)


def train_neural_network(network_architecture, learning_rate=1e-3, batch_size=100, epochs=100,
                         activation_fct=tf.nn.sigmoid):
    ''' Choosing the model Architecture '''

    autoencoder = DenoisingAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size,
                                      activation_fct=activation_fct)

    #autoencoder = StackedAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size,
     #                  activation_fct=activation_fct)

    #autoencoder = Autoencoder(n_layers=network_architecture['layers'],
     #                     transfer_function = tf.nn.softplus,
      #                    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

    # autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=trX.shape[1],
    #                                              n_z=int(50 * trX.shape[1] / 100),
    #                                             transfer_function=tf.nn.sigmoid,
    #                                            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    #                                           scale=0.001)

    #autoencoder = StackedSparseAutoencoder(n_input=trX.shape[1],
    #                                       network_architecture=network_architecture,
     #                                      n_z=network_architecture['n_z'],
      #                                     transfer_function=tf.nn.sigmoid,
       #                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        #                                   scale=0.001)

    n_batches = int(ceil(len(trX) / batch_size))
    train_batches = [_ for _ in generate_batches(trX.values, batch_size)]

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        '''
       train_accuracy_variable = tf.Variable(0.0, name="accuracy")

        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./tensorboard_summary/5")
        writer.add_graph(sess.graph) 
        '''

        writer_1 = tf.summary.FileWriter("./tensorboard_summary/602")

        log_var = tf.Variable(0.0)
        log_var2 = tf.Variable(0.0)
        tf.summary.scalar("accuracy", log_var)
        tf.summary.scalar("loss",log_var2)

        write_op = tf.summary.merge_all()

        for epoch in range(epochs):
            epoch_loss = 0
            avg_cost = 0.

            for i in range(n_batches):
                x_batch = train_batches[i]  # trX, trY = mnist.train.next_batch(batch)
                _, c = autoencoder.partial_fit(x_batch,corrupt_rate=corrupt_rate)  # sess.run([SAE.optimizer, SAE.cost], feed_dict={SAE.x: x_batch})

                epoch_loss += c
                # Compute average loss
                avg_cost += c / n_batches * batch_size

            print("Epoch:", (epoch + 1), ' Loss: ', epoch_loss, "cost=", "{:.9f}".format(avg_cost))

            x_hat = autoencoder.reconstructor(teX.values,corrupt_rate=corrupt_rate)
            train_accuracy = calculate_accuracy(teX.values, x_hat)
            print('Train accuracy: ', train_accuracy)
            print('')

            '''
            with tf.name_scope('accuracy'):
            tf.summary.scalar("Training Accuracy", train_accuracy_variable)

            summary = sess.run(merge_summary, feed_dict={train_accuracy_variable: train_accuracy})
            writer.add_summary(summary, epoch)
            writer.flush()
            
            '''
            if epoch % 10 == 0 and epoch != 0:
                summary = sess.run(write_op, {log_var: train_accuracy, log_var2: float(int(avg_cost))})
                writer_1.add_summary(summary, epoch)
                writer_1.flush()

        x_hat = autoencoder.reconstructor(teX.values,corrupt_rate=corrupt_rate)
        print(len(x_hat))
        print(len(teX))
        accuracy = calculate_accuracy(teX.values, x_hat)
        print('Accuracy: ', accuracy)

        # correct = tf.equal(tf.argmax(SAE.x_hat, 1), tf.argmax(SAE.x, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Old Accuracy :', accuracy.eval({SAE.x: trX}))

        """ Replace API Services """
        z = autoencoder.transformer(dataset.values,corrupt_rate=corrupt_rate)

        data_dict = dict()
        for i, api_name in enumerate(API_Names):
            if api_name in data_dict:
                pass #print(api_name)
            else:
                data_dict[api_name] = z[i]

        print(len(data_dict.keys()))

        prepare_mashup(data_dict)
        save_reduced_dimensions(data_dict)


layer1 = int(50 * trX.shape[1] / 100)
layer2 = int(50 * layer1 / 100)
layer3 = int(50 * layer2 / 100)
layer4 = int(50 * layer3 / 100)
layer5 = int(50 * layer4 / 100)
layer6 = int(50 * layer5 / 100)
layer7 = int(50 * layer6 / 100)
layer8 = int(50 * layer7 / 100)

print('[ ', trX.shape[1], ' ', layer1, ' ', layer2, ' ', layer3, ' ', layer4, ' ', layer5, ' ', layer6, ' ', layer7,
      ' ]')
network_architecture['layers'].append(trX.shape[1])
network_architecture['layers'].append(layer1)
network_architecture['layers'].append(layer2)
network_architecture['layers'].append(layer3)
network_architecture['layers'].append(layer5)
network_architecture['layers'].append(layer6)
#network_architecture['layers'].append(layer7)
network_architecture['layers'].append(10)


network_architecture['n_input'] = trX.shape[1]
network_architecture['n_z'] = 10
import time

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
train_neural_network(network_architecture, epochs=1000, batch_size=50, learning_rate=1e-3, activation_fct=tf.nn.sigmoid)

print("--- %s seconds ---" % (time.time() - start_time))


#print(API_Names)