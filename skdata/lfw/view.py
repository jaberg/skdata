# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@alum.mit.edu>

# License: Simplified BSD

from copy import deepcopy

import numpy as np

from skdata.utils import dotdict
from skdata.utils import ImgLoader
from skdata.larray import lmap

import dataset
DATASET_NAMES = ['Original', 'Funneled', 'Aligned']


class BaseView2(object):

    DATASET_CLASS = None

    def __init__(self):

        if self.DATASET_CLASS is None:
            raise NotImplementedError("This is an abstract class")

        # -- build/fetch dataset
        ds = self.DATASET_CLASS()
        ds.fetch()

        # -- load the pairs/labels txt file
        fname = ds.home('pairs.txt')
        pairs = np.loadtxt(fname, dtype=str, delimiter='\n')
        header = pairs[0].split()
        n_folds, n_pairs = map(int, header)
        n_pairs *= 2  # n_pairs 'same' + n_pairs 'different'

        # -- parse the folds
        folds = [[] for _ in xrange(n_folds)]
        i = 1
        for fold_i in xrange(n_folds):

            for _ in xrange(n_pairs):

                txt = pairs[i].split()
                # -- same
                if len(txt) == 3:
                    left = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[1]))
                    right = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[2]))
                    label = +1
                # -- different
                elif len(txt) == 4:
                    left = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[1]))
                    right = '%s/%s_%04d.jpg' % (txt[2], txt[2], int(txt[3]))
                    label = -1
                # --
                else:
                    raise RuntimeError("line not understood")
                folds[fold_i] += [(left, right, label)]

                i += 1

            assert len(folds[fold_i]) == n_pairs

        # -- lazy array helper function
        if ds.COLOR:
            loader = ImgLoader(ndim=3, dtype=np.float32, mode='RGB')
        else:
            loader = ImgLoader(ndim=2, dtype=np.float32, mode='L')

        def load_pair(pair):
            left_fname = ds.home('images', ds.IMAGE_SUBDIR, pair[0])
            left = loader(left_fname)
            right_fname = ds.home('images', ds.IMAGE_SUBDIR, pair[1])
            right = loader(right_fname)
            return np.array((left, right), dtype=np.float32)

        # -- for each fold build a lazy "split"
        splits = []
        all_x_fn = None
        all_y_fn = None
        for i, test_fold in enumerate(folds):

            # -- test
            test_fold = np.array(test_fold)

            test_x_fn = test_fold[:, :2]
            split_x_fn = deepcopy(test_x_fn)
            test_x = lmap(load_pair, test_x_fn)

            test_y_fn = test_fold[:, 2]
            split_y_fn = deepcopy(test_y_fn)
            test_y = lmap(load_pair, test_y_fn)

            if all_x_fn is None:
                all_x_fn = split_x_fn
                all_y_fn = split_y_fn
            else:
                all_x_fn = np.concatenate((all_x_fn, split_x_fn))
                all_y_fn = np.concatenate((all_y_fn, split_y_fn))

            # -- train (filenames)
            train_x_fn = None
            train_y_fn = None
            for j, train_fold in enumerate(folds):
                if j == i:
                    continue
                train_fold = np.array(train_fold)
                _train_x_fn = train_fold[:, :2]
                _train_y_fn = train_fold[:, 2]
                if train_x_fn is None:
                    train_x_fn = _train_x_fn
                    train_y_fn = _train_y_fn
                else:
                    train_x_fn = np.concatenate((train_x_fn, _train_x_fn))
                    train_y_fn = np.concatenate((train_y_fn, _train_y_fn))

            split_x_fn = np.concatenate((split_x_fn, train_x_fn))
            split_y_fn = np.concatenate((split_y_fn, train_y_fn))

            split_x = lmap(load_pair, train_x_fn)
            split_y = lmap(load_pair, train_y_fn)

            train_x = lmap(load_pair, train_x_fn)
            train_y = lmap(load_pair, train_y_fn)

            split = dotdict(
                x=split_x,
                y=split_y,
                train=dotdict(x=train_x, y=train_y),
                test=dotdict(x=test_x, y=test_y),
                )
            splits += [split]

        all_x = lmap(load_pair, all_x_fn)
        all_y = lmap(load_pair, all_y_fn)

        self.dataset = ds
        self.x = all_x
        self.y = all_y
        self.splits = splits


# -- Class Factory
for ds_name in DATASET_NAMES:
    name = '%sView2' % ds_name
    klass = type(name, (BaseView2, ),
                 BaseView2.__dict__.copy())
    klass.DATASET_CLASS = getattr(dataset, ds_name) 
    globals()[name] = klass
