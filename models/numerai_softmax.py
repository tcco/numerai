from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
sys.path.append(os.getcwd())
import tensorflow as tf
import util
from random import randint


def main(_):
    train_set, test_set = util.train_test_set()

    features = train_set.data.shape[1]
    outputs = 2
    train_data_set = util.Dataset(
        train_set.data,
        targets=train_set.target)
    test_data_set = util.Dataset(
        test_set.data,
        targets=test_set.target)
    print('Extracted train and test sets, building model...')
    # Softmax Model
    x = tf.placeholder(tf.float32, shape=[None, features])
    W = tf.Variable(tf.zeros([features, outputs]))
    b = tf.Variable(tf.zeros([outputs]))
    y = tf.matmul(x, W) + b
    # Loss
    y_ = tf.placeholder(tf.float32, [None, outputs])

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
    batch_size = 300
    for _ in range(1000):
        index = _ * batch_size
        batch_xs, batch_ys = train_data_set.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 100 == 0:
            print(' Training Step: {} \n Index: {}'.format(_, index))
            # print(y.eval({x: batch_xs}))

    print('Finished training, now testing on test set...')
    # Test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy on test set')
    print(sess.run(accuracy, feed_dict={x: test_data_set.data,
                                        y_: test_data_set.labels}))

    end = randint(10, test_set.data.shape[0])
    beg = end - 10
    print('Testing on random image (indices {}-{}) from test set...'.format(beg, end))
    feed_dict = {x: test_set.data[beg:end]}
    classification = sess.run(tf.argmax(y, 1), feed_dict=feed_dict)
    print(classification)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=util.DATA_DIR,
                        help='Directory for storing input data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
