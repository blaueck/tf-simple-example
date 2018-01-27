import os
# set tensorflow cpp log level. It is useful
# to diable some annoying log message, but sometime
#  may miss some useful imformation.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import importlib
import time

import numpy as np
import tensorflow as tf
from data_utils import load_data, split_data


def preprocess_for_eval(image, label):
    image = (tf.to_float(image) - 127.5) / 128.
    image = tf.transpose(image, [2, 0, 1])

    label = tf.to_int32(label)
    return image, label


def main(FLAGS):

    with tf.device('/cpu:0'), tf.name_scope('input'):

        # load dataset into main memory
        data, meta = load_data(
            FLAGS.dataset_root, FLAGS.dataset, is_training=False)

        # build tf_dataset for testing
        dataset = (tf.data.Dataset
            .from_tensor_slices(data)
            .map(preprocess_for_eval, 8)
            .batch(FLAGS.batch_size)
            .prefetch(4))

        # clean up and release memory
        del data

        # construct data iterator
        data_iterator = dataset.make_one_shot_iterator()

    # build the net
    model = importlib.import_module('models.{}'.format(FLAGS.model))
    net = model.Net(meta['n_class'])

    # get data from data iterator
    images, labels = data_iterator.get_next()

    # get logits
    logits = net(images, is_training=False)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    mean_loss, mean_loss_update_op = tf.metrics.mean(
        loss, name='mean_loss')

    prediction = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels, prediction, name='accuracy')

    # init op
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # saver
    saver = tf.train.Saver()

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # do initialization
    sess.run(init_op)

    # restore
    saver.restore(sess, FLAGS.checkpoint)

    # test loop 
    try:
        while True:
            sess.run([mean_loss_update_op, accuracy_update_op])
    except tf.errors.OutOfRangeError:
        pass

    loss, acc = sess.run([mean_loss, accuracy])
    print('[Test]Loss: {:.4f}, Acc: {:.2f}'.format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train dnn')
    parser.add_argument('--model', default='simple_cnn', help='model name')
    parser.add_argument('checkpoint', default='', help='snapshot path')
    parser.add_argument('--dataset', default='cifar10',
                        help='the training dataset')
    parser.add_argument(
        '--dataset_root', default='./data/cifar-10-batches-bin', help='dataset root')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args()
    main(args)
