import numpy as np
import tensorflow as tf
from lib.API_data_preprocessing import prepare_dataset
from lib.utils import generate_batches, calculate_accuracy

from deeplearning_models.variantional_autoencoder import VariationalAutoencoder

tf.set_random_seed(0)
np.random.seed(0)

dataset_name = 'final_API_data.csv'
dataset, X_train, X_test = prepare_dataset('./datasets/' + dataset_name)

API_Names = list(dataset.iloc[:, 0].values)
dataset = dataset.iloc[:, 1:]

teX = X_test.iloc[:, 1:]
trX = X_train.iloc[:, 1:]

# latent_space = int(80 * trX.shape[1] / 100)

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = len(trX)  # mnist.train.num_examples


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1,test_batches=[]):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 transfer_fct=tf.nn.softmax,
                                 batch_size=batch_size)

    total_batch = int(n_samples / batch_size)
    # train_batches = np.array_split(trX, total_batch)
    train_batches = [_ for _ in generate_batches(trX.values, batch_size)]


    with tf.Session() as sess:

        #Tensorboard Initialization

        sess.run(tf.global_variables_initializer())
        writer_1 = tf.summary.FileWriter("./tensorboard_summary/final_vae_501")

        log_var = tf.Variable(0.0)
        log_var2 = tf.Variable(0.0)
        tf.summary.scalar("accuracy", log_var)
        tf.summary.scalar("loss",log_var2)

        write_op = tf.summary.merge_all()

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_batches[i]
                # Fit training using batch data
                cost = vae.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(avg_cost))


            #Write to Tensorboard
            train_accuracy = 0.

            for i in range(len(test_batches)):
                x_bach = test_batches[i]
                x_reconstr = vae.reconstruct(x_bach)
                train_accuracy += calculate_accuracy(x_bach, x_reconstr)

            train_accuracy = train_accuracy / float(len(test_batches))
            print(train_accuracy)

            if epoch % 5 == 0 and epoch != 0:
                summary = sess.run(write_op, {log_var: train_accuracy, log_var2: float(int(avg_cost))})
                writer_1.add_summary(summary, epoch)
                writer_1.flush()

    return vae


layer1 = int(50 * trX.shape[1] / 100)
layer2 = int(50 * layer1 / 100)
layer3 = int(50 * layer2 / 100)
layer4 = int(50 * layer3 / 100)
layer5 = int(50 * layer4 / 100)
layer6 = int(50 * layer5 / 100)
layer7 = int(50 * layer6 / 100)

network_architecture = \
    dict(n_hidden_recog_1=layer1,  # 1st layer encoder neurons
         n_hidden_recog_2=layer2,  # 2nd layer encoder neurons
         n_hidden_recog_3=layer3,  # 3nd layer encoder neurons
         n_hidden_recog_4=layer4,  # 4nd layer encoder neurons
         n_hidden_recog_5=layer5,  # 5nd layer encoder neurons
         n_hidden_recog_6=layer6,  # 6nd layer encoder neurons

         n_hidden_gener_1=layer6,  # 1st layer decoder neurons
         n_hidden_gener_2=layer5,  # 2nd layer decoder neurons
         n_hidden_gener_3=layer4,  # 3nd layer decoder neurons
         n_hidden_gener_4=layer3,  # 4nd layer decoder neurons
         n_hidden_gener_5=layer2,  # 5nd layer decoder neurons
         n_hidden_gener_6=layer1,  # 6nd layer decoder neurons


         n_input=trX.shape[1],  # data input (832 attributs)
         n_z=10)  # dimensionality of latent space

import time

rest = teX.shape[0] % 100

test_batches = [_ for _ in generate_batches(teX.values[:-rest], 100)]


start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

vae = train(network_architecture, training_epochs=1000, learning_rate=1e-3,test_batches=test_batches)

print("--- %s seconds ---" % (time.time() - start_time))



avg_accuracy = 0.

for i in range(len(test_batches)):
    x_bach = test_batches[i]
    x_reconstr = vae.reconstruct(x_bach)
    avg_accuracy += calculate_accuracy(x_bach, x_reconstr)

avg_accuracy = avg_accuracy / float(len(test_batches))
print(avg_accuracy)



'''
correct = tf.equal(tf.argmax(x_reconstr, 1), tf.argmax(vae.x, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy:', accuracy.eval({vae.x: teX[:100]}))
'''
'''x_reconstr = vae.reconstruct(teX[:100])
correct = tf.equal(tf.argmax(x_reconstr, 1), tf.argmax(vae.x, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy:', accuracy.eval({vae.x: teX[:100]}))'''

'''





def train(network_architecture, learning_rate=0.001, batch_size=100, training_epochs=100, display_step=5):
    vae = VaraiontionalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batches = int(n_samples / batch_size)
        #train_batches = np.array_split(trX, total_batches)

        for i in range(total_batches):
            batch_xs = mnist.train.next_batch(batch_size) #train_batches[i]

            # fit training using Batch data
            cost = vae.pratical_fit(batch_xs)

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    return vae


network_architecture = \
    dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
         n_hidden_recog_2=500,  # 2nd layer encoder neurons
         n_hidden_gener_1=500,  # 1st layer decoder neurons
         n_hidden_gener_2=500,  # 2nd layer decoder neurons
         n_input=28*28,  # data input (size 1025)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=75)

'''
