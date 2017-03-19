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

import cPickle
import gzip


def extract():
    train_pd = pd.read_csv('numerai_training_data.csv', delimiter=',', header=0)
    tourn_pd = pd.read_csv('numerai_tournament_data.csv', delimiter=',', header=0)
    train = train_pd.as_matrix()
    tourn = tourn_pd.as_matrix()
    return train, tourn


def save_data():
    train, tourn = extract()
    with gzip.open('data/training.pklz', 'wb') as f:
        cPickle.dump(train, f, -1)

    with gzip.open('data/prediction.pklz', 'wb') as f:
        cPickle.dump(tourn, f, -1)


def import_data(filename):
    """ Example of how to save
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, -1)
    """
    with gzip.open(filename, 'rb') as f:
        return cPickle.load(f)

# DNNLinearRegression w/ Softmax outputs
import copy

wide_columns = []
deep_columns = []
continuous_columns = []
label_column = 'target'
for x in range(50):
    var = 'feature{}'.format(x + 1)
    vars()[var] = tf.contrib.layers.real_valued_column(var)
    continuous_columns.append(var)
    wide_columns.append(vars()[var])
    deep_columns.append(vars()[var])

m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir='ckpt/linear',
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[25, 25])

names = copy.copy(continuous_columns)
names.append(label_column)
df = pd.read_csv('numerai_training_data.csv',
                 names=names,
                 skipinitialspace=True,
                 skiprows=1)
msk = np.random.rand(len(df)) < .8
df_train = df[msk]
df_test = df[~msk]


pred_names = ['t_id']
pred_names.extend(names)
df_pred = pd.read_csv('numerai_tournament_data.csv',
                      names=pred_names,
                      skipinitialspace=True,
                      skiprows=1)
df_preds = df_pred[names]
df_tids = df_pred['t_id']

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_columns}
    feature_cols = dict(continuous_cols.items())
    label = tf.constant(df[label_column].values)
    return feature_cols, label


def all_input_fn():
    return input_fn(df)


def train_input_fn():
    return input_fn(df_train)


def test_input_fn():
    return input_fn(df_test)


def preds_input_fn():
    return input_fn(df_preds)


def predict():
    y = m.predict(input_fn=preds_input_fn, as_iterable=False)
    y_proba = m.predict_proba(input_fn=preds_input_fn, as_iterable=False)
    evaluation = m.evaluate(input_fn=all_input_fn, steps=1)
    return y, y_proba[:, 1], evaluation


def save_preds(pred_ones, t_id):
    np.savetxt('predictions.csv',
               zip(t_id, pred_ones),
               fmt='%d,%f',
               header='t_id,probability')

# FNN Softmax softmax_linear
"""

def restore(step, model_str='data/numerai'):
    meta = model_str + '/model.ckpt-' + step
    saver = tf.train.import_meta_graph('{}.meta'.format(meta))
    saver.restore(sess, meta)
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    names = []
    for o in ops:
        names.append(o.name)
    return graph, names

def tensor_restore(graph):
    h1w = graph.get_tensor_by_name('hidden1/weights:0')
    h1b = graph.get_tensor_by_name('hidden1/biases:0')
    h2w = graph.get_tensor_by_name('hidden2/weights:0')
    h2b = graph.get_tensor_by_name('hidden2/biases:0')
    smw = graph.get_tensor_by_name('softmax_linear/weights:0')
    smb = graph.get_tensor_by_name('softmax_linear/biases:0')
    return h1w, h1b, h2w, h2b, smw, smb


def inference(data, graph):
    h1w, h1b, h2w, h2b, smw, smb = tensor_restore(graph)
    hidden1 = tf.nn.relu(tf.matmul(data, h1w) + h1b)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, h2w) + h2b)
    logits = tf.nn.softmax(tf.matmul(hidden2, smw) + smb)
    return logits

prediction_set = util.prediction_set()
one_hot = False
preprocess = False
datasets = util.numerai_datasets(one_hot=one_hot, preprocess=preprocess)
train_set = datasets.train
test_set = datasets.test
classes = 2
features = 50
test_size = test_set.data.shape[0]

graph, names = restore('4066')
accuracy = graph.get_operation_by_name('xentropy_mean').outputs[0]
h1w, h1b, h2w, h2b, smw, smb = tensor_restore(graph)


data_pl, labels_pl = util.placeholder_inputs(
    test_set.data.shape[0],
    features,
    classes,
    one_hot=one_hot)
pred_data_pl = tf.placeholder(
    tf.float32,
    shape=prediction_set.data.shape,
    name='pred_data_pl')

logits = inference(data_pl, graph)
pred_logits = inference(pred_data_pl, graph)

data_feed = test_set.data
prediction_feed = prediction_set.data
label_feed = test_set.labels
target_feed = test_set.targets

sess.run(tf.global_variables_initializer())
feed_dict = {data_pl: data_feed, labels_pl: label_feed}
prediction_dict = {pred_data_pl: prediction_feed}
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('data/numerai')
saver.restore(sess, ckpt.model_checkpoint_path)
predictions = sess.run(pred_logits, prediction_dict)
pred_ones = predictions[:, 1]
tids = prediction_set.ids


preds = sess.run(logits, feed_dict)
argmax = sess.run(tf.argmax(preds, 1))
correct = float(np.sum(argmax == target_feed))
print 'Correct predictions: {} ({}%)'.format(correct, 100*correct / test_size)
loss = util.loss(logits, labels_pl, one_hot=one_hot)
loss_val = sess.run(loss, feed_dict=feed_dict)
print 'Loss: {}'.format(loss_val)


def evaluation(logits, labels):
    is_label_one = tf.cast(labels, dtype=tf.bool)
    is_label_zero = tf.logical_not(is_label_one)

    correct_prediction = tf.nn.in_top_k(logits, labels, 1, name='correct_eval')
    false_prediction = tf.logical_not(correct_prediction)

    tp = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction, is_label_one)))
    fp = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_zero)))
    tn = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction, is_label_zero)))
    fn = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_one)))
    return tp, fp, tn, fn


tp, fp, tn, fn = sess.run(evaluation(logits, labels_pl), feed_dict=feed_dict)
precision = float(tp) / float(tp+fp)
recall = float(tp) / float(tp+fn)
f1 = 2 * (precision * recall) / (precision+recall)
print '\t\tpredicted no, predicted yes \n\tactual no     {}    {}\n\tactual yes    {}    {}'.format(
    tn, fp, fn, tp)
print precision, recall, f1
"""
