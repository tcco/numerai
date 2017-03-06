"""A very simple Numerai classifier"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
sys.path.append(os.getcwd())

import time
import util
import numpy as np

import tensorflow as tf

FLAGS = None


def graph_inference(data, graph):
    h1w = graph.get_tensor_by_name('hidden1/weights:0')
    h1b = graph.get_tensor_by_name('hidden1/biases:0')
    h2w = graph.get_tensor_by_name('hidden2/weights:0')
    h2b = graph.get_tensor_by_name('hidden2/biases:0')
    smw = graph.get_tensor_by_name('softmax_linear/weights:0')
    smb = graph.get_tensor_by_name('softmax_linear/biases:0')
    hidden1 = tf.nn.relu(tf.matmul(data, h1w) + h1b)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, h2w) + h2b)
    logits = tf.nn.softmax(tf.matmul(hidden2, smw) + smb)
    return logits


def analyze(sess,
            test_logits,
            test_labels,
            test_feed_dict,
            test_loss,
            test_samples):
    preds = sess.run(test_logits, feed_dict=test_feed_dict)
    argmax = sess.run(tf.argmax(preds, 1))
    correct = float(np.sum(argmax == test_labels))
    accuracy = 100 * correct / test_samples
    test_loss_value = sess.run(test_loss, feed_dict=test_feed_dict)
    return test_loss_value, accuracy


def run_training():
    # Import data
    datasets = util.numerai_datasets(one_hot=FLAGS.one_hot)
    features = datasets.features
    classes = datasets.classes
    test_samples = datasets.test.data.shape[0]
    test_data = datasets.test.data
    test_labels = datasets.test.labels
    print('Extracted train and test sets, building model...')

    with tf.Graph().as_default() as graph:
        # Create the model
        print('Creating data and labels placeholders...\n')
        data_placeholder, labels_placeholder = util.placeholder_inputs(FLAGS.batch_size,
                                                                       num_features=features,
                                                                       num_classes=classes,
                                                                       one_hot=FLAGS.one_hot)
        test_data_pl, test_labels_pl = util.placeholder_inputs(
            test_samples,
            num_features=features,
            num_classes=classes,
            one_hot=FLAGS.one_hot)
        test_feed_dict = {test_data_pl: test_data, test_labels_pl: test_labels}
        print('Logits using data placholder...\n')
        logits = util.inference(data_placeholder,
                                FLAGS.hidden1,
                                FLAGS.hidden2,
                                num_classes=classes,
                                num_features=features)
        test_logits = graph_inference(test_data_pl, graph)

        print('Loss using labels placholder...\n')
        loss = util.loss(logits, labels_placeholder, one_hot=FLAGS.one_hot)
        test_loss = util.loss(test_logits, test_labels_pl,
                              one_hot=FLAGS.one_hot)

        print('Add training operation...\n')
        train_op = util.training(
            loss, FLAGS.learning_rate, FLAGS.decay, FLAGS.max_steps)

        print('Build summary Tensor based on collections...\n')
        summary = tf.summary.merge_all()
        print('Add initializer Op, saver, and session for writing checkpoints...\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        if FLAGS.pickup:
            ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        best_loss = 1.0
        subset_loss = 1.0
        best_accuracy = 0.0

        print('Training...\n')
        start_time = time.time()
        for step in xrange(FLAGS.max_steps):
            feed_dict = util.fill_feed_dict(datasets.train,
                                            data_placeholder,
                                            labels_placeholder,
                                            FLAGS.batch_size)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            if loss_value < subset_loss:
                subset_loss = loss_value
                # Real best loss and accuracy and step
                l, a = analyze(sess, test_logits, test_labels,
                               test_feed_dict, test_loss, test_samples)
                if l < best_loss:
                    best_step = step
                    best_loss = l
                    best_accuracy = a
                    best_sess = sess

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                # Print status to stdout.
                print('Step %d: loss = %.6f accuracy = %.4f (%.4f sec)' %
                      (step, best_loss, best_accuracy, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if best_loss < FLAGS.loss_goal or (step + 1) == FLAGS.max_steps:
                print('best step %d: loss = %.6f accuracy = %.4f' %
                      (best_step, best_loss, best_accuracy))
                if FLAGS.pickup:
                    tf.gfile.DeleteRecursively(FLAGS.log_dir)
                    tf.gfile.MakeDirs(FLAGS.log_dir)
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(best_sess, checkpoint_file, global_step=best_step)
                return


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        if FLAGS.pickup:
            pass
        else:
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
            tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pickup',
        type=bool,
        default=False,
        help='Pickup where checkpoint leftoff.'
    )
    parser.add_argument(
        '--decay',
        type=bool,
        default=False,
        help='Decay learning rate.'
    )
    parser.add_argument(
        '--one_hot',
        type=bool,
        default=False,
        help='Determines sparse vs proba output.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.5,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--loss_goal',
        type=float,
        default=0.69,
        help='Goal of loss we are trying to find less than.'
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
        default=25,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=25,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='data/numerai',
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
