from pathlib import Path
import pickle
import numpy as np
import gzip
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


DIR = Path(__file__).resolve().parent


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype=np.float32)
    data_y = np.asarray(data_y, dtype=np.int64)
    return data_x, data_y


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]  # The shape is same as .gz


def preprocess():
    data = [
        load_data(DIR / 'noisymnist_view1.gz'),
        load_data(DIR / 'noisymnist_view2.gz')
    ]

    train_x, train_y = {i: view[0][0] for i, view in enumerate(data)}, data[0][0][1]  # [view][train/valid/test][x/y]
    valid_x, valid_y = {i: view[1][0] for i, view in enumerate(data)}, data[0][1][1]
    test_x, test_y = {i: view[2][0] for i, view in enumerate(data)}, data[0][2][1]

    print('Saving processed multi-view dataset ...')
    pickle.dump([train_x, train_y], open(DIR / 'train.pkl', 'wb'))
    pickle.dump([valid_x, valid_y], open(DIR / 'valid.pkl', 'wb'))
    pickle.dump([test_x, test_y], open(DIR / 'test.pkl', 'wb'))


if __name__ == '__main__':
    preprocess()
