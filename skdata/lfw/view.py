# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@alum.mit.edu>

# License: Simplified BSD

from copy import deepcopy

import numpy as np

from skdata.utils import dotdict
from skdata.utils import ImgLoader
from skdata.larray import lmap

import dataset


def paths_labels(pairs):
    """
    Returns tensor of shape (n_folds, n_labels * n_pairs) of recarrays with
    ['lpath', 'rpath', 'label'] fields.
    """
    n_folds, n_labels, n_pairs, n_per_pair = pairs.shape
    assert n_per_pair == 2

    def foo(l, r):
        (lname, lnum), (rname, rnum)
        lpath = '%s/%s_%04d.jpg' % (lname, lname, lnum)
        rpath = '%s/%s_%04d.jpg' % (rname, rname, rnum)
        label = 1 if lname == rname else -1
        return lpath, rpath, label

    rval = np.recarray(n_folds * n_labels * n_pairs,
            dtype=np.dtype([
                ('lpath', 'S' + 3 * dataset.NAMELEN),
                ('rpath', 'S' + 3 * dataset.NAMELEN),
                ('label', np.int8)]))
    rval[:] = map(foo, pairs.reshape((-1, 2)))
    return rval.reshape((n_folds, n_labels * n_pairs))


def sorted_paths(paths_labels):
    """
    Return a sorted sequence of all paths that occur in paths_labels
    """
    paths = list(set(
        list(paths_labels['lpath'].flatten())
        + list(paths_labels['rpath'].flatten())))
    paths.sort()
    return paths


def paths_labels_lookup(paths_labels, path_list):
    """
    `paths_labels` is a ndarray of recarrays with string paths
    replace the path strings with integers of where to find paths in the
    pathlist.

    Return recarray has fields ['lpathidx', 'rpathidx', 'label'].
    """
    rval = np.recarray(paths_labels.shape,
            dtype=np.dtype([
                ('lpathidx', np.int32),
                ('rpathidx', np.int32),
                ('label', np.int8)]))
    rval['lpathidx'] = np.searchsorted(path_list, path_labels['lpath'])
    rval['rpathidx'] = np.searchsorted(path_list, path_labels['rpath'])
    rval['label'] = path_labels['label']
    return rval


class FullProtocol(object):

    DATASET_CLASS = None

    def __init__(self):
        if self.DATASET_CLASS is None:
            raise NotImplementedError("This is an abstract class")

        # -- build/fetch dataset
        ds = self.DATASET_CLASS()

        paths_labels_dev_train = paths_labels(ds.pairsDevTrain)
        paths_labels_dev_test = paths_labels(ds.pairsDevTest)
        paths_labels_view2 = paths_labels(ds.pairsView2)
        all_paths_labels = np.concatenate([
            paths_labels_dev_train.flatten(),
            paths_labels_dev_test.flatten(),
            paths_labels_view2.flatten()])

        self.img_paths = sorted_paths(all_paths_labels)

        def lookup(pairs):
            return paths_labels_lookup(paths_labels(pairs), self.img_paths)

        self.dev_train = lookup(ds.pairsDevTrain)
        self.dev_test = lookup(ds.pairsDevTest)
        self.view2 = lookup(ds.pairsView2)

        # -- lazy array helper function
        if ds.COLOR:
            loader = ImgLoader(ndim=3, dtype=np.float32, mode='RGB')
        else:
            loader = ImgLoader(ndim=2, dtype=np.float32, mode='L')

        self.img_pixels = lmap(loader, self.img_paths)
        self.paths_labels_dev_train = paths_labels_dev_train
        self.paths_labels_dev_test = paths_labels_dev_test
        self.paths_labels_view2 = paths_labels_view2


    @property
    def protocol(self):

        def task(obj, name):
            return Task('image_match_indexed',
                    lidx=obj['lpathidx'],
                    ridx=obj['rpathidx'],
                    y=obj['label'],
                    images=self.img_pixels,
                    name=name)

        model = BestModelByCrossValidation(
                Split(
                    task(self.dev_train[0], 'devTrain'),
                    task(self.dev_test[0], 'devTest')))

        v2_scores = []

        for i, v2i_tst in enumerate(view2):
            v2i_tst = task(self.view2[i], 'view2_test_%i' % i)
            v2i_trn = Task('image_match_indexed',
                    lidx=np.concatenate([self.view2[j]['lpathidx']
                        for j in range(10) if j != i]),
                    ridx=np.concatenate([self.view2[j]['rpathidx']
                        for j in range(10) if j != i]),
                    y=np.concatenate([self.view2[j]['label']
                        for j in range(10) if j != i]),
                    images=self.img_pixels,
                    name='view2_train_%i' % i,
                    )
            v2i_model = RetrainClassifier(model, v2i_trn)
            v2_scores.append(Score(v2i_model, v2i_tst,
                name='view2 test error %i' % i))

        return Average(v2_scores, name='final view2 test error')


class Original(FullProtocol):
    DATASET_CLASS = dataset.Original


class Funneled(FullProtocol):
    DATASET_CLASS = dataset.Funneled


class Aligned(FullProtocol):
    DATASET_CLASS = dataset.Aligned


class BaseView2(FullProtocol):

    def __init__(self):
        FullProtocol.__init__(self)

        if self.DATASET_CLASS is None:
            raise NotImplementedError("This is an abstract class")

        # -- build/fetch dataset
        ds = self.DATASET_CLASS()
        folds = ds.pairsView2

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

            test_y = test_fold[:, 2].astype(int)
            split_y = deepcopy(test_y)

            if all_x_fn is None:
                all_x_fn = split_x_fn
                all_y = split_y
            else:
                all_x_fn = np.concatenate((all_x_fn, split_x_fn))
                all_y = np.concatenate((all_y, split_y))

            # -- train (filenames)
            train_x_fn = None
            train_y = None
            for j, train_fold in enumerate(folds):
                if j == i:
                    continue
                train_fold = np.array(train_fold)
                _train_x_fn = train_fold[:, :2]
                _train_y = train_fold[:, 2].astype(int)
                if train_x_fn is None:
                    train_x_fn = _train_x_fn
                    train_y = _train_y
                else:
                    train_x_fn = np.concatenate((train_x_fn, _train_x_fn))
                    train_y = np.concatenate((train_y, _train_y))

            split_x_fn = np.concatenate((split_x_fn, train_x_fn))
            split_y = np.concatenate((split_y, train_y))

            split_x = lmap(load_pair, train_x_fn)
            train_x = lmap(load_pair, train_x_fn)

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


class OriginalView2(BaseView2):
    DATASET_CLASS = dataset.Original


class FunneledView2(BaseView2):
    DATASET_CLASS = dataset.Funneled


class AlignedView2(BaseView2):
    DATASET_CLASS = dataset.Aligned
