# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD

import numpy as np
from scipy import io

from skdata.utils import dotdict
from skdata.larray import lmap

from dataset import SVHNCroppedDigits


class SVHNCroppedDigitsStratifiedKFoldView1(object):

    def __init__(self, k=10):

        from sklearn.cross_validation import StratifiedKFold

        ds = SVHNCroppedDigits()

        mat = io.loadmat(ds.meta['train']['filename'])
        x = np.rollaxis(mat['X'], -1)
        y = mat['y'].ravel()

        cv = StratifiedKFold(y, k=k)

        def split_func(*args):
            trn_idx, tst_idx = args[0][0]
            trn_x, trn_y = x[trn_idx], y[trn_idx]
            tst_x, tst_y = x[tst_idx], y[tst_idx]
            split = dotdict(
                train=dotdict(x=trn_x, y=trn_y),
                test=dotdict(x=tst_x, y=tst_y),
                )
            return split

        splits = lmap(split_func, zip(cv))
        self.dataset = ds
        self.splits = splits


class SVHNCroppedDigitsView2(object):

    def __init__(self):

        ds = SVHNCroppedDigits()

        train_mat = io.loadmat(ds.meta['train']['filename'])
        train_x = np.rollaxis(train_mat['X'], -1).astype(np.float32)
        train_y = train_mat['y'].ravel().astype(np.float32)

        test_mat = io.loadmat(ds.meta['test']['filename'])
        test_x = np.rollaxis(test_mat['X'], -1).astype(np.float32)
        test_y = test_mat['y'].ravel().astype(np.float32)

        split = dotdict()
        split['train'] = dotdict(x=train_x, y=train_y)
        split['test'] = dotdict(x=test_x, y=test_y)

        self.dataset = ds
        self.splits = [split]
