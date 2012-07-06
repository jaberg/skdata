import numpy as np
from scipy import io

from skdata.utils import dotdict

from dataset import SVHNCroppedDigits


class SVHNCroppedDigitsView2(object):

    def __init__(self):

        ds = SVHNCroppedDigits()

        print "Loading train data..."
        train_mat = io.loadmat(ds.meta['train']['filename'])
        train_x = np.rollaxis(train_mat['X'], -1).astype(np.float32)
        train_y = train_mat['y'].ravel().astype(np.float32)

        print "Loading test data..."
        test_mat = io.loadmat(ds.meta['test']['filename'])
        test_x = np.rollaxis(test_mat['X'], -1).astype(np.float32)
        test_y = test_mat['y'].ravel().astype(np.float32)

        split = dotdict()
        split['train'] = dotdict(x=train_x, y=train_y)
        split['test'] = dotdict(x=test_x, y=test_y)
        self.splits = [split]
