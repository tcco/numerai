"""
Handles importing many of the core classes, methods and models of the project.

:copyright: (c) 2016 Pinn Technologies, Inc.
:license: All rights reserved
"""

import click
click.secho('>>> import os',
            fg='white')
import os  # NOQA
import sys  # NOQA
sys.path.append(os.getcwd())

# Models and stuff
click.secho('>>> import util',
            fg='white')
import util  # NOQA

click.secho('>>> import models',
            fg='white')
import models  # NOQA

# Tensorflow
click.secho('>>> import tensorflow as tf',
            fg='white')
import tensorflow as tf  # NOQA
tf.logging.set_verbosity(tf.logging.INFO)

click.secho('>>> from tensorflow.examples.tutorials.mnist import input_data',
            fg='white')
from tensorflow.examples.tutorials.mnist import input_data  # NOQA

click.secho('>>> mnist = input_data.read_data_sets(MNIST_DATA, one_hot=True)',
            fg='white')
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)  # NOQA

click.secho('>>> sess = tf.InteractiveSession()',
            fg='white')
sess = tf.InteractiveSession()

# Other
click.secho('>>> from sklearn import preprocessing',
            fg='white')
from sklearn import preprocessing  # NOQA

click.secho('>>> import numpy as np',
            fg='white')
import numpy as np  # NOQA

click.secho('>>> import pandas as pd',
            fg='white')
import pandas as pd  # NOQA

click.secho('>>> Dataset = collections.namedtuple(Dataset, [data, target])',
            fg='white')
import collections
Dataset = collections.namedtuple('Dataset', ['data', 'target'])

# Data
train_set, test_set = util.train_test_set()
hidden_units = [10, 20, 10]
n_classes = 2
model_dir = "tmp/numerai"

num_features = train_set.data.shape[1]
FEATURES = ['feature-{}'.format(i+1) for i in range(num_features)]
LABEL = 'target'
COLUMNS = FEATURES + [LABEL]
feature_columns = [tf.contrib.layers.real_valued_column(k)
                   for k in FEATURES]
feature_cols = feature_columns

training_frame_data = np.append(
    train_set.data,
    train_set.target.reshape(train_set.data.shape[0], 1),
    axis=1)

test_frame_data = np.append(
    test_set.data,
    test_set.target.reshape(test_set.data.shape[0], 1),
    axis=1)

training_frame = pd.DataFrame(training_frame_data, columns=COLUMNS)
testing_frame = pd.DataFrame(test_frame_data, columns=COLUMNS)


def example_classifier():
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        model_dir=model_dir)
    return classifier


def input_fn(data_set):
    """
    Parameters
    ----------
    data_set : pandas.DataFrame
        DataFrame containing Tensor initialized data

    Returns
    -------
    feature : dict
        Key/value feature/tensor of feature.
    label : tf.Tensor
    """
    feature_cols = {k: tf.constant(data_set[k].values,
                                   shape=[data_set[k].size, 1])
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values,
                         shape=[data_set[LABEL].size, 1])
    return feature_cols, labels


# classifier = example_classifier()


# def fi(frame, steps=0):
#     classifier.fit(input_fn=lambda: input_fn(frame), steps=steps)


# def ev(frame, steps=1):
#     return classifier.evaluate(input_fn=lambda: input_fn(frame), steps=steps)

# Testing

def import_data(filename):
    """ Example of how to save
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, -1)
    """
    import cPickle
    import gzip
    with gzip.open(filename, 'rb') as f:
        return cPickle.load(f)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
