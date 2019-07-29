import tensorflow as tf
from lib.utils import xavier_init


class StackedAutoencoder(object):
    def __init__(self, network_architecture, learning_rate=1e-3, batch_size=100, epochs=100,
                 activation_fct=tf.nn.sigmoid):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.activation_fct = activation_fct

        self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])

        self.create_network()

        self._create_loss_optimizer()

        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

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
            output = tf.add(tf.matmul(current_layer, W), b)
            output = self.activation_fct(output)

            current_layer = output

        self.z = current_layer

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

        self.x_hat = current_layer

    def _create_loss_optimizer(self):
        self.cost = tf.reduce_sum(tf.pow(self.x - self.x_hat, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.session.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.session.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.session.run(self.z, feed_dict={self.x: x})
        return z

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.session.run((self.optimizer, self.cost), feed_dict={self.x: X})

        return opt, cost
