import tensorflow as tf  # NOQA
import numpy as np  # NOQA
import collections  # NOQA
import math  # NOQA
from sklearn import preprocessing  # NOQA
from tensorflow.python.framework import dtypes  # NOQA


Datasets = collections.namedtuple('Datasets', ['train', 'test', 'features', 'classes'])
TrainTestSet = collections.namedtuple('TrainTestSet', ['data', 'target'])
PredictionSet = collections.namedtuple('PredictionSet', ['data', 'sample', 'ids'])


def import_data(filename):
    import cPickle
    import gzip
    with gzip.open(filename, 'rb') as f:
        return cPickle.load(f)


def training_data():
    return import_data('data/training.pklz')


def train_test_set(preprocess=True,
                   test_percentage=.2):
    training = training_data()
    if preprocess:
        ############# Range
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(training[:, :-1])
        ############# Standardize
        # data = preprocessing.scale(training[:, :-1])
        ############# Normalize
        # data = preprocessing.normalize(training[:, :-1], norm='l2')
    else:
        data = training[:, :-1]
    target = training[:, -1]
    test_size = int(training.shape[0]*test_percentage)
    train_set = TrainTestSet(
        data=data[0:-test_size],
        target=target[0:-test_size])
    test_set = TrainTestSet(
        data=data[-test_size:],
        target=target[-test_size:])
    return train_set, test_set


def train_set(preprocess=True):
    training = training_data()
    if preprocess:
        ############# Range
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(training[:, :-1])
        ############# Standardize
        # data = preprocessing.scale(training[:, :-1])
        ############# Normalize
        # data = preprocessing.normalize(training[:, :-1], norm='l2')
    else:
        data = training[:, :-1]
    target = training[:, -1]
    train_set = TrainTestSet(
        data=data,
        target=target)
    return train_set


def prediction_data(preprocess=True,
                    size=19):
    prediction = import_data('data/prediction.pklz')
    if preprocess:
        data = preprocessing.scale(prediction[:, 1:])
    else:
        data = prediction[:, 1:]
    sample = data[0:size]
    ids = prediction[:, 0]
    prediction_set = PredictionSet(data=data,
                                   sample=sample,
                                   ids=ids)
    return prediction_set


class Dataset(object):

    def __init__(self, data, labels=np.array([]), dtype=dtypes.float32, targets=None):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid dtype, expected uint8 or float32')

        assert data.shape[0] == labels.shape[0] or data.shape[0] == targets.shape[0], (
            'data.shape: %s labels.shape: %s' % (data.shape, labels.shape))

        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._targets = targets
        if targets is not None:
            self.target_to_label(targets)

    def __repr__(self):
        return ("  ++++++++++++++++++++++++++++\n" +
                "  {} <{}>\n".format(self.__class__.__name__, hex(id(self))) +
                "  data.shape:       {} \n".format(self._data.shape) +
                "  labels.shape:     {} \n".format(self._labels.shape) +
                "  num_examples:     {} \n".format(self._num_examples) +
                "  epochs_completed: {}".format(self._epochs_completed))

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def targets(self):
        if self._targets is None:
            return self._labels
        return self._targets

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def target_to_label(self, target):
        """Target column to Labels, assure targets are int scalar from 0-n"""
        assert len(target.shape) == 1, 'Targets are not in need of reshaping?'
        print('One_hot equivalent! Converting targets to labels...\n')
        classes = int(np.ptp(target, axis=0) + 1)
        for t in target:
            tmp = self._labels
            z = np.zeros(classes)
            z[int(t)] = 1.0
            if self._labels.shape[0] == 0:
                self._labels = np.hstack((tmp, z))
            else:
                self._labels = np.vstack((tmp, z))

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle and begin with first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
        # Go to next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest of examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return (np.concatenate((data_rest_part, data_new_part), axis=0),
                    np.concatenate((labels_rest_part, labels_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]



def numerai_datasets(one_hot=True):
    train_set, test_set = train_test_set()
    features = train_set.data.shape[1]
    classes = int(np.ptp(train_set.target, axis=0) + 1)
    if one_hot:
        train_dataset = Dataset(train_set.data, targets=train_set.target)
        test_dataset = Dataset(test_set.data, targets=test_set.target)
    else:
        train_dataset = Dataset(train_set.data, labels=train_set.target)
        test_dataset = Dataset(test_set.data, labels=test_set.target)
    return Datasets(train_dataset, test_dataset, features, classes)