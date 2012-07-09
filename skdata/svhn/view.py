# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD

import numpy as np
from scipy import io

from ..utils import dotdict
from ..larray import lmap
from ..dslang import Task

from dataset import CroppedDigits


class CroppedDigitsStratifiedKFoldView1(object):

    def __init__(self, k=10):

        from sklearn.cross_validation import StratifiedKFold

        ds = CroppedDigits()

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


class CroppedDigitsView2(object):

    def __init__(self, x_dtype='float32', n_train=None):

        ds = CroppedDigits()

        train_mat = io.loadmat(ds.meta['train']['filename'])
        train_x = np.rollaxis(train_mat['X'], -1).astype(np.float32)
        train_y = train_mat['y'].ravel().astype(np.float32)
        train = Task(x=train_x, y=train_y)

        test_mat = io.loadmat(ds.meta['test']['filename'])
        test_x = np.rollaxis(test_mat['X'], -1).astype(np.float32)
        test_y = test_mat['y'].ravel().astype(np.float32)
        test = Task(x=test_x, y=test_y)

        split = dotdict()
        split['train'] = train
        split['test'] = test

        self.dataset = ds
        self.splits = [split]
        self.train = train
        self.test = test

        # XXX missing protocol

