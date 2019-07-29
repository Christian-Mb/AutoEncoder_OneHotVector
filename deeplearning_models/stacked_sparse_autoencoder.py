import tensorflow as tf
import numpy as np
import math

from lib.utils import xavier_init


class StackedSparseAutoencoder(object):
    def __init__(self, n_input, network_architecture, n_z, transfer_function=tf.nn.sigmoid,
                 optimizer=tf.train.AdamOptimizer(),scale=0.1, learning_rate=1e-3):

        self.n_input = n_input
        self.n_z = n_z
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.activation_fct = transfer_function

        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.sparsity_level = np.repeat([0.05], self.n_z).astype(np.float32)
        self.sparse_reg = 0.0

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.create_network()

        self._create_loss_optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def create_network(self):
        # n_input = self.network_architecture['n_input']

        current_layer = self.x
        Encoders = dict(weights=[], biases=[])

        for i, n_output in enumerate(self.network_architecture['layers'][1:]):
            n_input = int(current_layer.get_shape()[1])
            W = tf.Variable(xavier_init(n_input, n_output))
            b = tf.Variable(tf.zeros([n_output]))
            print('weight ', W.shape)
            print('biase ', b.shape)

            Encoders['weights'].append(W)
            Encoders['biases'].append(b)

            print('encoder', W.shape)
            print('encoder', b.shape)

            output = self.activation_fct(
                tf.add(tf.matmul(current_layer + self.training_scale * tf.random_normal((n_input,)), W), b))

            # output = tf.add(tf.matmul(current_layer, W), b)
            # output = self.activation_fct(output)

            current_layer = output

        self.hidden = current_layer

        Decoders = dict(weights=[], biases=[])
        Decoders['weights'].extend(Encoders['weights'])
        Decoders['biases'].extend(Encoders['biases'])
        Decoders['weights'].reverse()
        Decoders['biases'].reverse()

        for i, n_output in enumerate(self.network_architecture['layers'][:-1][::-1]):
            W = tf.transpose(Decoders['weights'][i])
            b = tf.Variable(tf.zeros([n_output]))
            print('weight decoder', W.shape)
            print('biases decoder', b.shape)

            output = tf.add(tf.matmul(current_layer, W), b)
            output = self.activation_fct(output)
            current_layer = output

        self.reconstruction = current_layer

    def _create_loss_optimizer(self):
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) + self.sparse_reg \
                                                                                                 * self.kl_divergence(
            self.sparsity_level, self.hidden)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

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
