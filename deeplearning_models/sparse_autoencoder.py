import tensorflow as tf
import numpy as np
import math

from lib.utils import xavier_init


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_z, transfer_function=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_z
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.sparsity_level = np.repeat([0.05], self.n_hidden).astype(np.float32)
        self.sparse_reg = 0.0

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']),
                                           self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) + self.sparse_reg \
                                                                                                 * self.kl_divergence(
            self.sparsity_level, self.hidden)

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.transpose(all_weights['w1'])) #tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale
                                                                          })
        return opt, cost

    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat))

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale
                                                   })

    def transformer(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale
                                                     })

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstructor(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale
                                                             })

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

'''

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)




#X_train = mnist.train.images
#X_test = mnist.test.images
dataset_name = 'API_data.csv'
trX, teX = prepare_dataset('./datasets/' + dataset_name)

#n_samples = int(mnist.train.num_examples)

training_epochs = 100
batch_size = 50
display_step = 1
n_z = int(50 * trX.shape[1] / 100)

import time

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))



n_samples = int(math.ceil(len(trX) / batch_size))  # (int(mnist.train.num_examples / batch))
train_batches = [_ for _ in generate_batches(trX.values, batch_size)]  # np.array_split(trX,n_batches)

for epoch in range(training_epochs):
    avg_cost = 0.

    for i in range(len(train_batches)):
        batch_xs = train_batches[i] #get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", avg_cost)

print("Total cost: " + str(autoencoder.calc_total_cost(teX)))

print("--- %s seconds ---" % (time.time() - start_time))

x_reconstr = autoencoder.reconstruct(teX.values)
print(calculate_accuracy(teX.values,x_reconstr))

'''