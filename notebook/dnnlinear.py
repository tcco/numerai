
# coding: utf-8

# In[5]:

import os  # NOQA
import sys  # NOQA
sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import util
import argparse

wide_columns = []
deep_columns = []
continuous_columns = []
label_column = 'target'

for x in range(50):
    var = 'feature{}'.format(x+1)
    vars()[var] = tf.contrib.layers.real_valued_column(var)
    continuous_columns.append(var)
    wide_columns.append(vars()[var])
    deep_columns.append(vars()[var])

names = copy.copy(continuous_columns)
names.append(label_column)
pred_names = ['t_id']
pred_names.extend(names)

df = pd.read_csv('numerai_training_data.csv',
                 names=names,
                 skipinitialspace=True,
                 skiprows=1)
msk = np.random.rand(len(df)) < .8
df_train = df[msk]
df_test = df[~msk]

df_pred = pd.read_csv('numerai_tournament_data.csv',
                      names=pred_names,
                      skipinitialspace=True,
                      skiprows=1)
df_preds = df_pred[names]
df_tids = df_pred['t_id']

# Model
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir='ckpt/linear',
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[25, 25])

print 'DNNLinearRegression Setup Complete'


# In[6]:

# Method Definitions


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


def train(steps=5000):
    m.fit(input_fn=all_input_fn, steps=steps)

    
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

print 'Method definitions complete'


# In[ ]:


def run_training():
    print 'Training Deep Neural Network w/ Linear Regression + Softmax Output'
    import time
    start = time.time()
    train(steps=FLAGS.steps)
    stop = time.time()
    y, y_proba, evaluation = predict()
    util.logger(evaluation)
    save_preds(y_proba, df_tids.as_matrix())
    print '\n\nTotal training time: {}\n\n'.format(stop - start)
    

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=int,
        default=5000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='ckpt/linear',
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# In[ ]:



