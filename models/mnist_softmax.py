# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from random import randint
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def runner():
    return main('tmp/mnist')


def main(_):
    # Import data
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
    print('Extracted train and test sets, building model...')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    print('Analyzing based on cross entropy...')
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    print('Adding loss to summary')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    print('Building interactive session...')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    print('Training...')
    index = 0
    batch_size = 100
    for _ in range(1000):
        index = _ * batch_size
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 100 == 0:
            print(' Training Step: {} \n Index: {}'.format(_, index))
            # print(y.eval({x: batch_xs}))

    print('Finished training, now testing on test set...')
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy on test set')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    end = randint(10, mnist.test.images.shape[0])
    beg = end - 10
    print('Testing on random image (indices {}-{}) from test set...'.format(beg, end))
    feed_dict = {x: mnist.test.images[beg:end]}
    classification = sess.run(tf.argmax(y, 1), feed_dict=feed_dict)
    print(classification)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='tmp/mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
