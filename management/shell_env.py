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

click.secho('>>> mnist = input_data.read_data_sets(data/MNIST_DATA, one_hot=True)',
            fg='white')
mnist = input_data.read_data_sets('data/MNIST_DATA', one_hot=True)  # NOQA

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


def restore(step, model_str='data/numerai/model.ckpt-'):
    meta = model_str + step
    saver = tf.train.import_meta_graph('{}.meta'.format(meta))
    saver.restore(sess, meta)
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    names = []
    for o in ops:
        names.append(o.name)
    return graph, names


datasets = util.numerai_datasets(one_hot=True)
train_set = datasets.train
test_set = datasets.test
classes = 2
features = 50
'''Return from meta model'''
graph, names = restore('2397')
accuracy = graph.get_operation_by_name('xentropy_mean').outputs[0]
data_pl = graph.get_operation_by_name('data_pl').outputs[0]
labels_pl = graph.get_operation_by_name('labels_pl').outputs[0]
test_size = int(data_pl.shape[0])
'''Rebuild logits'''
'''REBUILD BY HAND INFERENCE FUNCTION'''
logits = util.inference(data_pl, 25, 25, classes, features)
'''Pred / Accuracy data'''
from random import randint
beg = randint(0, test_set.data.shape[0]-test_size)
end = beg + test_size
print 'Testing {} - {}'.format(beg, end)
data_feed = test_set.data[beg:end]
label_feed = test_set.labels[beg:end]
target_feed = test_set.targets[beg:end]
'''Predictions'''
prediction = tf.nn.softmax(logits)
sess.run(tf.global_variables_initializer())
feed_dict = {data_pl: data_feed}
preds = sess.run(prediction, feed_dict=feed_dict)
argmax = sess.run(tf.argmax(preds, 1))
print 'Correct predictions: {}'.format(np.sum(argmax == target_feed))
'''Accuracy'''
feed_dict = {data_pl: data_feed, labels_pl: label_feed}
acc = sess.run(accuracy, feed_dict=feed_dict)
print 'Meta accuracy {}'.format(acc)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_pl, 1))
accurate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc2 = sess.run(accurate, feed_dict=feed_dict)
print 'Rebuilt accuracy {}'.format(acc2)


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
