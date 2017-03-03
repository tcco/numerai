"""
Utility folder with machine learning convenience.
"""

from data import Dataset
from data import train_set
from data import train_test_set
from data import numerai_datasets

from fnn import placeholder_inputs
from fnn import inference
from fnn import loss
from fnn import training
from fnn import evaluation
from fnn import fill_feed_dict
from fnn import do_eval

NUMERAI_SOFTMAX = 'data/numerai/softmax'