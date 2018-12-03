import os
import glob
import gzip

import numpy as np
from scipy.io import loadmat


def _glob(pattern):

    if isinstance(pattern, str):
        files = glob.glob(pattern)
    elif isinstance(pattern, list):
        files = []
        for p in pattern:
            files.extend(glob.glob(pattern))
    else:
        raise TypeError('wrong argument type.')

    return files


def get_cifar10(files):

    images_splits = []
    labels_splits = []
    n_pixel = 32 * 32 * 3

    for f in files:
        buffer = np.fromfile(f, dtype='uint8')
        buffer = buffer.reshape(-1, n_pixel+1)

        images = buffer[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = buffer[:, 0]

        images_splits.append(images)
        labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def get_cifar100(files):

    images_splits = []
    labels_splits = []
    n_pixel = 32 * 32 * 3

    for f in files:
        buffer = np.fromfile(f, dtype='uint8')
        buffer = buffer.reshape(-1, n_pixel+2)

        images = buffer[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = buffer[:, 1]

        images_splits.append(images)
        labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def get_mnist(image_files, label_files):

    images_splits = []
    labels_splits = []

    for i_f, l_f in zip(image_files, label_files):

        with gzip.open(i_f, 'rb') as f:
            images = np.frombuffer(f.read(), dtype='uint8', offset=16)
            images = images.reshape(-1, 28, 28, 1)
            images_splits.append(images)

        with gzip.open(l_f, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype='uint8', offset=8)
            labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def get_svhn(data_file):

    data = loadmat(data_file)
    images = data['X'].transpose(3, 0, 1, 2)
    labels = data['y'].reshape(-1)

    # map label 10 to 0
    labels[labels == 10] = 0

    return images, labels


def load_data(root, dataset, is_training):

    if dataset == 'cifar10':

        if is_training:
            pattern = os.path.join(root, 'data_batch*.bin')
        else:
            pattern = os.path.join(root, 'test_batch.bin')

        files = _glob(pattern)
        assert files, 'no file is matched.'

        data = get_cifar10(files)
        meta = {'n_class': 10}

    elif dataset == 'cifar100':

        if is_training:
            pattern = os.path.join(root, 'train.bin')
        else:
            pattern = os.path.join(root, 'test.bin')

        files = _glob(pattern)
        assert files, 'no file is matched.'

        data = get_cifar100(pattern)
        meta = {'n_class': 100}

    elif dataset == 'mnist':

        if is_training:
            img_pattern, label_pattern = [
                os.path.join(root, fn)
                for fn in ['train-images-idx3-ubyte.gz',
                           'train-labels-idx1-ubyte.gz']]
        else:
            img_pattern, label_pattern = [
                os.path.join(root, fn)
                for fn in ['t10k-images-idx3-ubyte.gz',
                           't10k-labels-idx1-ubyte.gz']]
        
        img_files, label_files = _glob(img_pattern), _glob(label_pattern)
        assert img_files, 'no image file is matched.'
        assert label_files, 'no label file is matched.'

        data = get_mnist(img_files, label_files)
        meta = {'n_class': 10}
    
    elif dataset == 'svhn':

        if is_training:
            pattern = os.path.join(root, 'train_32x32.mat')
        else:
            pattern = os.path.join(root, 'test_32x32.mat')
        
        data_file = _glob(pattern)[0]
        data = get_svhn(data_file)
        meta = {'n_class': 10}

    else:
        raise ValueError('%s is not supported.' % dataset)
    
    meta['name'] = dataset
    return data, meta


def split_data(data, rate, shuffle=True):

    images, labels = data
    N = images.shape[0]
    split_point = int(N * rate)

    if shuffle:
        idx = np.random.permutation(N)
        images, labels = images[idx], labels[idx]

    train_data = images[split_point:], labels[split_point:]
    val_data = images[:split_point], labels[:split_point]

    return train_data, val_data
