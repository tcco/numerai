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

# Tensorflow
click.secho('>>> import tensorflow as tf',
            fg='white')
import tensorflow as tf  # NOQA

click.secho('>>> from tensorflow.examples.tutorials.mnist import input_data',
            fg='white')
from tensorflow.examples.tutorials.mnist import input_data  # NOQA

click.secho('>>> mnist = input_data.read_data_sets(MNIST_DATA, one_hot=True)',
            fg='white')
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)  # NOQA

click.secho('>>> sess = tf.InteractiveSession()',
            fg='white')
sess = tf.InteractiveSession()

# Data


def import_data(filename):
    """ Example of how to save
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, -1)
    """
    import cPickle
    import gzip
    with gzip.open(filename, 'rb') as f:
        return cPickle.load(f)

training = import_data('training.pklz')
prediction = import_data('prediction.pklz')

# Testing


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
