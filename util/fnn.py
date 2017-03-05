import tensorflow as tf  # NOQA
import numpy as np  # NOQA
import collections  # NOQA
import math  # NOQA
from sklearn import preprocessing  # NOQA
from tensorflow.python.framework import dtypes  # NOQA


def placeholder_inputs(batch_size, num_features, num_classes=1, one_hot=True):
  data_shape = (batch_size, num_features)
  data_placeholder = tf.placeholder(tf.float32, shape=data_shape, name='data_pl')
  labels_shape = (batch_size)
  if one_hot:
    labels_shape = (batch_size, num_classes)
  labels_placeholder = tf.placeholder(tf.int32, shape=labels_shape, name='labels_pl')
  return data_placeholder, labels_placeholder


def inference(data, hidden1_units, hidden2_units, num_classes, num_features):
	"""Build the model up to where it may be used for inference.

  	Returns:
    	softmax_linear: Output tensor with the computed logits.
  	"""
	with tf.name_scope('hidden1'):
		weights = tf.Variable(
			tf.truncated_normal([num_features, hidden1_units],
								stddev=1.0 / math.sqrt(float(num_features))),
			name='weights')
		biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
		hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)

	with tf.name_scope('hidden2'):
		weights = tf.Variable(
			tf.truncated_normal([hidden1_units, hidden2_units],
								stddev=1.0 / math.sqrt(float(hidden1_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(
			tf.truncated_normal([hidden2_units, num_classes],
								stddev=1.0 / math.sqrt(float(hidden2_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([num_classes]), name='biases')
		logits = tf.matmul(hidden2, weights) + biases
	return logits


def loss(logits, labels, one_hot=True):
  """Calculates the loss from the logits and the labels.

  Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size].

  Returns:
	loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  if one_hot:
    print('One_hot, using softmax probability distribution...\n')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  else:
    print('Not one_hot, using sparse softmax for discrete labels...\n')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels, one_hot=True):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  if not one_hot:
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
  else:
    pass


def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  images_feed, labels_feed = data_set.next_batch(batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            batch_size):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               batch_size)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
