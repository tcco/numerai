from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
import tensorflow as tf
import util
from random import randint


FLAGS = None


def main(_):
    datasets = util.numerai_datasets(one_hot=True)
    features = datasets.features
    classes = datasets.classes

    print('Extracted train and test sets, building model...')
    # Softmax Model
    x = tf.placeholder(tf.float32, shape=[None, features])
    W = tf.Variable(tf.zeros([features, classes]))
    b = tf.Variable(tf.zeros([classes]))
    y = tf.matmul(x, W) + b
    # Loss
    y_ = tf.placeholder(tf.float32, [None, classes])

    # Cross Entropy
    print('Analyzing based on cross entropy...')
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Build Interactive Session
    print('Building interactive session...')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    print('Training...')
    index = 0
    batch_size = FLAGS.batch_size
    for _ in range(FLAGS.max_steps):
        index = _ * batch_size
        batch_xs, batch_ys = datasets.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 100 == 0:
            print(' Training Step: {} \n Index: {}'.format(_, index))
            # print(y.eval({x: batch_xs}))

    print('Finished training, now testing on test set...')
    # Test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy on test set')
    print(sess.run(accuracy, feed_dict={x: datasets.test.data,
                                        y_: datasets.test.labels}))

    end = randint(10, datasets.test.data.shape[0])
    beg = end - 10
    print('Testing on random image (indices {}-{}) from test set...'.format(beg, end))
    feed_dict = {x: datasets.test.data[beg:end]}
    print(sess.run(tf.argmax(y, 1), feed_dict=feed_dict))
    print('Testing on ONE random image (indice {}) from test set...'.format(beg))
    a = tf.constant(np.array(datasets.test.data[beg]))
    print(sess.run(tf.nn.softmax(a)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=util.NUMERAI_SOFTMAX,
                        help='Directory for storing input data.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Size of batches, try to divide evenly into dataset sizes.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Numer of steps to run trainer.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
