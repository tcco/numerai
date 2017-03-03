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

import os
import argparse
import sys
sys.path.append(os.getcwd())
import util
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def run_training():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    features = mnist.train.images.shape[1]
    classes = 10
    print('Extracted train and test sets, building model...')

    with tf.Graph().as_default():
        # Create the model
        print('Creating data and labels placeholders...\n')
        data_placeholder, labels_placeholder = util.placeholder_inputs(FLAGS.batch_size,
                                                                       num_features=features)
        print('Logits using data placholder...\n')
        logits = util.inference(data_placeholder,
                                FLAGS.hidden1,
                                FLAGS.hidden2,
                                num_classes=classes,
                                num_features=features)

        print('Loss using labels placholder...\n')
        loss = util.loss(logits, labels_placeholder)

        print('Add training operation...\n')
        train_op = util.training(loss, FLAGS.learning_rate)

        print('Evaluation of logits compared to labels...\n')
        eval_correct = util.evaluation(logits, labels_placeholder)

        print('Build summary Tensor based on collections...\n')
        summary = tf.summary.merge_all()
        print('Add initializer Op, saver, and session for writing checkpoints...\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(init)

        best_loss = 1.0
        best_step = 0
        how_many = 0

        print('Training...\n')
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = util.fill_feed_dict(mnist.train,
                                            data_placeholder,
                                            labels_placeholder,
                                            FLAGS.batch_size)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time

            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                how_many += 1

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.4f (%.5f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                util.do_eval(
                    sess,
                    eval_correct,
                    data_placeholder,
                    labels_placeholder,
                    mnist.train,
                    FLAGS.batch_size)
                print('Test Data Eval:')
                util.do_eval(
                    sess,
                    eval_correct,
                    data_placeholder,
                    labels_placeholder,
                    mnist.test,
                    FLAGS.batch_size)
        print('How many best? %d Best loss value = %.4f at step %d' % (how_many, best_loss, best_step))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2500,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='data/MNIST_DATA',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='data/mnist',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
