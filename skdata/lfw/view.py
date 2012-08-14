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

    def foo(lr):
        (lname, lnum), (rname, rnum) = lr
        lpath = '%s/%s_%04d.jpg' % (lname, lname, lnum)
        rpath = '%s/%s_%04d.jpg' % (rname, rname, rnum)
        assert len(lpath) < (3 * dataset.NAMELEN)
        assert len(rpath) < (3 * dataset.NAMELEN)
        label = 1 if lname == rname else -1
        return lpath, rpath, label

    rval = np.recarray(n_folds * n_labels * n_pairs,
            dtype=np.dtype([
                ('lpath', 'S' + str(3 * dataset.NAMELEN)),
                ('rpath', 'S' + str(3 * dataset.NAMELEN)),
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
    rval['lpathidx'] = np.searchsorted(path_list, paths_labels['lpath'])
    rval['rpathidx'] = np.searchsorted(path_list, paths_labels['rpath'])
    rval['label'] = paths_labels['label']
    return rval


class FullProtocol(object):

    """
    image_pixels:
        lazy array of grey or rgb images as pixels, all images in
        dataset.

    view2: integer recarray of shape (10, 600) whose fields are:
        'lpathidx': index of left image in image_pixels
        'rpathidx': index of right image in image_pixels
        'label':    -1 or 1

    """

    DATASET_CLASS = None

    def __init__(self, x_dtype='uint8', x_height=250, x_width=250):
        if self.DATASET_CLASS is None:
            raise NotImplementedError("This is an abstract class")

        # -- build/fetch dataset
        ds = self.DATASET_CLASS()
        ds.meta

        paths_labels_dev_train = paths_labels(ds.pairsDevTrain)
        paths_labels_dev_test = paths_labels(ds.pairsDevTest)
        paths_labels_view2 = paths_labels(ds.pairsView2)
        all_paths_labels = np.concatenate([
            paths_labels_dev_train.flatten(),
            paths_labels_dev_test.flatten(),
            paths_labels_view2.flatten()])

        rel_paths = sorted_paths(all_paths_labels)

        self.image_paths = [ds.home('images', ds.IMAGE_SUBDIR, pth)
                for pth in rel_paths]

        def lookup(pairs):
            return paths_labels_lookup(paths_labels(pairs), rel_paths)

        self.dev_train = lookup(ds.pairsDevTrain)
        self.dev_test = lookup(ds.pairsDevTest)
        self.view2 = lookup(ds.pairsView2)

        # -- lazy array helper function
        if ds.COLOR:
            ndim, mode, shape = (3, 'RGB', (x_height, x_width, 3))
        else:
            ndim, mode, shape = (2, 'L', (x_height, x_width))
        loader = ImgLoader(ndim=ndim, dtype=x_dtype, mode=mode, shape=shape)

        self.dataset = ds
        self.image_pixels = lmap(loader, self.image_paths)
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
                    images=self.image_pixels,
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
                    images=self.image_pixels,
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

    """
    self.dataset - a dataset.BaseLFW subclass instance
    self.x all image pairs in view2
    self.y all image pair labels in view2
    self.splits : list of 10 View2 splits, each one has
        splits[i].x : all of the image pairs in View2
        splits[i].y : all labels of splits[i].x
        splits[i].train.x : subset of splits[i].x
        splits[i].train.y : subset of splits[i].x
        splits[i].test.x : subset of splits[i].x
        splits[i].test.y : subset of splits[i].x
    """

    def load_pair(self, idxpair):
        lidx, ridx, label = idxpair

        # XXX
        # WTF why does loading this as a numpy int32 cause it
        # to try to load a path '/' whereas int() make it load the right path?
        l = self.image_pixels[int(lidx)]
        r = self.image_pixels[int(ridx)]
        return np.asarray([l, r])

    def __init__(self, *args, **kwargs):
        FullProtocol.__init__(self, *args, **kwargs)
        view2 = self.view2

        all_x = lmap(self.load_pair, view2.flatten())
        all_y = self.view2.flatten()['label']
        splits = []
        for fold_i, test_fold in enumerate(view2):

            # -- test
            test_x = lmap(self.load_pair, test_fold)
            test_y = test_fold['label']

            train_x = lmap(self.load_pair,
                    np.concatenate([
                        fold
                        for fold_j, fold in enumerate(view2)
                        if fold_j != fold_i]))
            train_y = np.concatenate([
                fold['label']
                for fold_j, fold in enumerate(view2)
                if fold_j != fold_i])

            splits.append(
                    dotdict(
                        x=all_x,
                        y=all_y,
                        train=dotdict(x=train_x, y=train_y),
                        test=dotdict(x=test_x, y=test_y),
                        )
                    )

        self.x = all_x
        self.y = all_y
        self.splits = splits

    @property
    def protocol(self):
        raise NotImplementedError()


class OriginalView2(BaseView2):
    DATASET_CLASS = dataset.Original


class FunneledView2(BaseView2):
    DATASET_CLASS = dataset.Funneled


class AlignedView2(BaseView2):
    DATASET_CLASS = dataset.Aligned
