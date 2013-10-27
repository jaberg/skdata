# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD

import numpy as np
from scipy import io

from sklearn.cross_validation import StratifiedShuffleSplit

from ..utils import dotdict
from ..larray import lmap
from ..dslang import Task

from dataset import CroppedDigits


class CroppedDigitsStratifiedKFoldView1(object):

    def __init__(self, k=10):

        from sklearn.cross_validation import StratifiedKFold

        ds = CroppedDigits(need_extra=False)

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

    def __init__(self, x_dtype=np.float32):

        ds = CroppedDigits()

        train_mat = io.loadmat(ds.meta['train']['filename'])
        train_x = np.rollaxis(train_mat['X'], -1).astype(x_dtype)
        train_y = train_mat['y'].ravel().astype(np.int32)
        train = Task(x=train_x, y=train_y)

        test_mat = io.loadmat(ds.meta['test']['filename'])
        test_x = np.rollaxis(test_mat['X'], -1).astype(x_dtype)
        test_y = test_mat['y'].ravel().astype(np.int32)
        test = Task(x=test_x, y=test_y)

        split = dotdict()
        split['train'] = train
        split['test'] = test

        self.dataset = ds
        self.splits = [split]
        self.train = train
        self.test = test


class CroppedDigitsSupervised(object):
    max_n_train = 73257
    max_n_test = 26032

    def __init__(self, x_dtype=np.float32,
                 n_train=max_n_train,
                 n_valid=0,
                 n_test=max_n_test,
                 shuffle_seed=123,
                 channel_major=False,
                ):

        assert n_train + n_valid <= self.max_n_train
        assert n_test <= self.max_n_test

        if shuffle_seed:
            rng = np.random.RandomState(shuffle_seed)
        else:
            rng = None

        ds = CroppedDigits(need_extra=False)

        train_mat = io.loadmat(ds.meta['train']['filename'])
        train_x = np.rollaxis(train_mat['X'], -1)
        train_y = train_mat['y'].ravel().astype(np.int32)
        assert len(train_x) == self.max_n_train

        test_mat = io.loadmat(ds.meta['test']['filename'])
        test_x = np.rollaxis(test_mat['X'], -1)
        test_y = test_mat['y'].ravel().astype(np.int32)
        assert len(test_x) == self.max_n_test

        # train_x and test_x are piled together here partly because I haven't
        # tested downstream code's robustness to Tasks with different values
        # for `all_images`, which should be fine in principle.
        all_x = np.concatenate((train_x, test_x), axis=0)
        all_y = np.concatenate((train_y, test_y), axis=0)
        del train_x, test_x

        assert all_x.dtype == np.uint8
        if 'float' in str(x_dtype):
            all_x = all_x.astype(x_dtype) / 255.0
        else:
            all_x = all_x.astype(x_dtype)

        all_y -= 1
        assert all_y.min() == 0
        assert all_y.max() == 9

        if channel_major:
            all_x = all_x.transpose(0, 3, 1, 2).copy()
            assert all_x.shape[1] == 3
        else:
            assert all_x.shape[3] == 3

        if n_train < self.max_n_train:
            ((fit_idxs, val_idxs),) = StratifiedShuffleSplit(
                y=all_y[:self.max_n_train],
                n_iterations=1,
                test_size=n_valid,
                train_size=n_train,
                indices=True,
                random_state=rng)
        else:
            fit_idxs = np.arange(self.max_n_train)
            val_idxs = np.arange(0)

        sel_idxs = np.concatenate([fit_idxs, val_idxs])

        if n_test < self.max_n_test:
            ((ign_idxs, tst_idxs),) = StratifiedShuffleSplit(
                y=all_y[self.max_n_train:],
                n_iterations=1,
                test_size=n_test,
                indices=True,
                random_state=rng)
            tst_idxs += self.max_n_train
            del ign_idxs
        else:
            tst_idxs = np.arange(self.max_n_train, len(all_x))

        self.dataset = ds
        self.n_classes = 10
        self.fit_idxs = fit_idxs
        self.val_idxs = val_idxs
        self.sel_idxs = sel_idxs
        self.tst_idxs = tst_idxs
        self.all_x = all_x
        self.all_y = all_y

    def protocol(self, algo):
        for _ in self.protocol_iter(algo):
            pass
        return algo

    def protocol_iter(self, algo):

        def task(name, idxs):
            return Task(
                'indexed_image_classification',
                name=name,
                idxs=idxs,
                all_images=self.all_x,
                all_labels=self.all_y,
                n_classes=self.n_classes)

        task_fit = task('fit', self.fit_idxs)
        task_val = task('val', self.val_idxs)
        task_sel = task('sel', self.sel_idxs)
        task_tst = task('tst', self.tst_idxs)

        if len(self.val_idxs):
            model1 = algo.best_model(train=task_fit, valid=task_val)
            yield ('model validation complete', model1)

        model2 = algo.best_model(train=task_sel)
        algo.loss(model2, task_tst)
        yield ('model testing complete', model2)

