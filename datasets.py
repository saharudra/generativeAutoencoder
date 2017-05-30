'''
This file is just for loading the various kinds of datasets in a particular
format.
This is taken from iwae code at 
https://github.com/yburda/iwae
'''


import config

import theano
import theano.tensor as T
import numpy as np

import struct
import os


class DatasetTheano():
    def __init__(self, train_data, test_data, n_used_for_validation, shuffle=False, shuffle_seed=123):
        self.data = {}

        if shuffle:
            permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
            train_data = train_data[permutation]

        self.data['train'] = train_data[:-n_used_for_validation]
        self.data['validation'] = train_data[-n_used_for_validation:]
        self.data['test'] = test_data

        for subdataset in ['train', 'validation', 'test']:
            self.data[subdataset] = theano.shared(value=self.data[subdataset].astype(theano.config.floatX))

    def minibatchIindex_minibatch_size(self, index, minibatch_size, subdataset='train', **kwargs):
        return self.data[subdataset][index*minibatch_size: (index+1)*minibatch_size]

    def get_data_dim(self):
        return self.data['train'].get_value(borrow=True).shape[1]

    def get_n_examples(self, subdataset):
        return self.data[subdataset].get_value(borrow=True).shape[0]

    def get_train_bias_np(self):
        return -np.log(1./np.clip(self.get_train_mean_np(), 0.001, 0.999)-1.)\
                .astype(theano.config.floatX)

    def get_train_mean_np(self):
        return np.mean(self.data['train'].get_value(), axis=0)[None, :].astype(theano.config.floatX)


class BinarizedDatasetTheano():
    def __init__(self, dataset):
        self.data = {}
        for subdataset in dataset.data:
            self.data[subdataset] = dataset.data[subdataset]

    def minibatchIindex_minibatch_size(self, index, minibatch_size, srng, subdataset):
        data = self.data[subdataset][index*minibatch_size: (index+1)*minibatch_size]
        binary_data = T.cast(T.le(srng.uniform(data.shape), data), data.dtype)
        return binary_data

    def get_data_dim(self):
        return self.data['train'].get_value(borrow=True).shape[1]

    def get_n_examples(self, subdataset):
        return self.data[subdataset].get_value(borrow=True).shape[0]

    def get_train_bias_np(self):
        return -np.log(1./np.clip(self.get_train_mean_np(), 0.001, 0.999)-1.)\
                .astype(theano.config.floatX)

    def get_train_mean_np(self):
        return np.mean(self.data['train'].get_value(), axis=0)[None, :].astype(theano.config.floatX)

def binarized_mnist(n_validation=400):
    def load_mnist_images_np(imgs_filename):
        with open(imgs_filename, 'rb') as f:
            f.seek(4)
            nimages, rows, cols = struct.unpack('>iii', f.read(12))
            dim = rows*cols

            images = np.fromfile(f, dtype=np.dtype(np.ubyte))
            images = (images/255.0).astype('float32').reshape((nimages, dim))

        return images

    train_data = load_mnist_images_np(
        os.path.join(config.DATASETS_DIR, 'MNIST', 'train-images-idx3-ubyte'))
    test_data = load_mnist_images_np(
        os.path.join(config.DATASETS_DIR, 'MNIST', 't10k-images-idx3-ubyte'))

    return BinarizedDatasetTheano(DatasetTheano(train_data, test_data, n_validation, shuffle=False))

