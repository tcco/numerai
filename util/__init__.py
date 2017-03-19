"""
Utility folder with machine learning convenience.
"""

from data import Dataset  # NOQA
from data import train_set  # NOQA
from data import train_test_set  # NOQA
from data import prediction_set  # NOQA
from data import numerai_datasets  # NOQA

from fnn import placeholder_inputs  # NOQA
from fnn import inference  # NOQA
from fnn import loss  # NOQA
from fnn import training  # NOQA
from fnn import evaluation  # NOQA
from fnn import fill_feed_dict  # NOQA
from fnn import do_eval  # NOQA

import json
import datetime

NUMERAI_SOFTMAX = 'data/numerai/softmax'


def logger(message):
    import inspect
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    func = inspect.currentframe().f_back.f_code
    packet = {'time': ts,
              'name': func.co_name,
              'file_name': func.co_filename,
              'mesage': str(message)}
    with open('data/loggy.logs', 'a+') as l:
        l.write('\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n')
        json.dump(packet, l, indent=4, sort_keys=True)
        l.write('\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n')
