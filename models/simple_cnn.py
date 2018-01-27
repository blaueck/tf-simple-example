import tensorflow as tf
from .base import BaseNet


class Net(BaseNet):

    def __init__(self, n_class, weight_decay=0.0001, name='simple_cnn'):
        super().__init__(name)

        self.weight_decay = weight_decay
        self.data_format = 'channels_first'
        self.n_class = n_class

    def call(self, images, is_training):
        w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(2),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
        }
        conv_args = {
            'data_format': self.data_format,
            'padding': 'same',
            **w_args
        }
        fc_args = {
            **w_args
        }

        net = tf.layers.conv2d(images, 32, 3, 2, name='conv1', **conv_args)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 64, 3, 2, name='conv2', **conv_args)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 128, 3, 2, name='conv3', **conv_args)
        net = tf.nn.relu(net)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, self.n_class, name='fc1', **fc_args)
        return net
