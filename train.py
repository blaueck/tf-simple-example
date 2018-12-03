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


def preprocess_for_train(random_flip=True):
    def func(image, label):
        shape = image.get_shape().as_list()
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
        image = tf.random_crop(image, shape)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        image = (tf.to_float(image) - 127.5) / 128.
        image = tf.transpose(image, [2, 0, 1])

        label = tf.to_int64(label)
        return image, label
    return func


def preprocess_for_eval(image, label):
    image = (tf.to_float(image) - 127.5) / 128.
    image = tf.transpose(image, [2, 0, 1])

    label = tf.to_int64(label)
    return image, label


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time
        self.counter += 1

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.


class LRManager:

    def __init__(self, boundaries, values):
        self.boundaries = boundaries
        self.values = values

    def get(self, epoch):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                return v
        return self.values[-1]


def main(FLAGS):

    # set seed
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    with tf.device('/cpu:0'), tf.name_scope('input'):

        # load dataset into main memory
        data, meta = load_data(
            FLAGS.dataset_root, FLAGS.dataset, is_training=True)
        train_data, val_data = split_data(data, FLAGS.validate_rate)

        # build tf_dataset for training
        train_dataset = (tf.data.Dataset
            .from_tensor_slices(train_data)
            .map(preprocess_for_train(args.dataset not in ['mnist', 'svhn']), 8)
            .shuffle(10000, seed=FLAGS.seed)
            .batch(FLAGS.batch_size)
            .prefetch(1))

        # build tf_dataset for val
        val_dataset = (tf.data.Dataset
            .from_tensor_slices(val_data)
            .map(preprocess_for_eval, 8)
            .batch(FLAGS.batch_size)
            .prefetch(1))

        # clean up and release memory
        del data, train_data, val_data

        # construct data iterator
        data_iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes)

        # construct iterator initializer for training and validation
        train_data_init = data_iterator.make_initializer(train_dataset)
        val_data_init = data_iterator.make_initializer(val_dataset)


    # define useful scalars
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    tf.summary.scalar('lr', learning_rate)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.train.create_global_step()

    # define optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # build the net
    model = importlib.import_module('models.{}'.format(FLAGS.model))
    net = model.Net(meta['n_class'], FLAGS.weight_decay)

    # get data from data iterator
    images, labels = data_iterator.get_next()
    tf.summary.image('images', tf.transpose(images, [0, 2, 3, 1]))

    # get logits
    logits = net(images, is_training)
    tf.summary.histogram('logits', logits)

    # summary variable defined in net
    for w in net.global_variables:
        tf.summary.histogram(w.name, w)


    with tf.name_scope('losses'):
        # compute loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        # compute l2 regularization
        l2_reg = tf.losses.get_regularization_loss()


    with tf.name_scope('metrics') as scope:

        mean_loss, mean_loss_update_op = tf.metrics.mean(
            loss, name='mean_loss')

        prediction = tf.argmax(logits, axis=1)
        accuracy, accuracy_update_op = tf.metrics.accuracy(
            labels, prediction, name='accuracy')

        reset_metrics = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope))
        metrics_update_op = tf.group(mean_loss_update_op, accuracy_update_op)

        # collect metric summary alone, because it need to
        # summary after metrics update
        metric_summary = [
            tf.summary.scalar('loss', mean_loss, collections=[]),
            tf.summary.scalar('accuracy', accuracy, collections=[])]

    # compute grad
    grads_and_vars = optimizer.compute_gradients(loss + l2_reg)

    # summary grads
    for g, v in grads_and_vars:
        tf.summary.histogram(v.name + '/grad', g)

    # run train_op and update_op together
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(train_op, *update_ops)

    # build summary
    train_summary_str = tf.summary.merge_all()
    metric_summary_str = tf.summary.merge(metric_summary)

    # init op
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # prepare for the logdir
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)

    # saver
    saver = tf.train.Saver(max_to_keep=FLAGS.n_epoch)

    # summary writer
    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'train'),
        tf.get_default_graph())
    val_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'val'),
        tf.get_default_graph())

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # do initialization
    sess.run(init_op)

    # restore
    if FLAGS.restore:
        saver.restore(sess, FLAGS.restore)

    lr_boundaries = list(map(int, FLAGS.boundaries.split(',')))
    lr_values = list(map(float, FLAGS.values.split(',')))
    lr_manager = LRManager(lr_boundaries, lr_values)
    time_meter = TimeMeter()

    # start to train
    for e in range(FLAGS.n_epoch):
        print('-' * 40)
        print('Epoch: {:d}'.format(e))

        # training loop
        try:
            i = 0
            sess.run([train_data_init, reset_metrics])
            while True:
                
                lr = lr_manager.get(e)
                fetch = [train_summary_str] if i % FLAGS.log_every == 0 else []

                time_meter.start()
                result = sess.run(
                    [train_op, metrics_update_op] + fetch,
                    {learning_rate: lr, is_training: True})
                time_meter.stop()

                if i % FLAGS.log_every == 0:
                    # fetch summary str
                    t_summary = result[-1]
                    t_metric_summary = sess.run(metric_summary_str)

                    t_loss, t_acc = sess.run([mean_loss, accuracy])
                    sess.run(reset_metrics)

                    spd = FLAGS.batch_size / time_meter.get()
                    time_meter.reset()

                    print('Iter: {:d}, LR: {:g}, Loss: {:.4f}, Acc: {:.2f}, Spd: {:.2f} i/s'
                          .format(i, lr, t_loss, t_acc, spd))

                    train_writer.add_summary(
                        t_summary, global_step=sess.run(global_step))
                    train_writer.add_summary(
                        t_metric_summary, global_step=sess.run(global_step))

                i += 1
        except tf.errors.OutOfRangeError:
            pass

        # save checkpoint
        saver.save(sess, '{}/{}'.format(FLAGS.logdir, FLAGS.model),
                   global_step=sess.run(global_step), write_meta_graph=False)

        # val loop
        try:
            sess.run([val_data_init, reset_metrics])
            while True:
                sess.run([metrics_update_op], {is_training: False})
        except tf.errors.OutOfRangeError:
            pass

        v_loss, v_acc = sess.run([mean_loss, accuracy])
        print('[VAL]Loss: {:.4f}, Acc: {:.2f}'.format(v_loss, v_acc))

        val_writer.add_summary(sess.run(metric_summary_str),
                               global_step=sess.run(global_step))

    print('-' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train dnn')
    parser.add_argument('--dataset', default='svhn',
                        help='the training dataset')
    parser.add_argument(
        '--dataset_root', default='./data/svhn', help='dataset root')
    parser.add_argument(
        '--logdir', default='log/simple_cnn', help='log directory')
    parser.add_argument('--restore', default='', help='snapshot path')
    parser.add_argument('--validate_rate', default=0.1,
                        type=float, help='validate split rate')

    parser.add_argument('--model', default='simple_cnn', help='model name')

    parser.add_argument('--n_epoch', default=70,
                        type=int, help='number of epoch')
    parser.add_argument('--weight_decay', default=0.0001,
                        type=float, help='weight decay rate')
    parser.add_argument('--boundaries', default='30,50,60',
                        help='learning rate boundaries')
    parser.add_argument(
        '--values', default='1e-2,1e-2,1e-3,1e-4', help='learning rate values')

    parser.add_argument('--log_every', default=100, type=int,
                        help='display and log frequency')
    parser.add_argument('--seed', default=0, type=float, help='random seed')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args()
    main(args)
