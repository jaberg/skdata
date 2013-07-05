"""
Experiment views on the Iris data set.

"""

import numpy as np
from sklearn import cross_validation

from .dataset import Iris
from ..base import Task
from ..utils import int_labels


class KfoldClassification(object):
    """
    Access train/test splits for K-fold cross-validation as follows:

    >>> self.splits[k].train.x
    >>> self.splits[k].train.y
    >>> self.splits[k].test.x
    >>> self.splits[k].test.y

    """

    def __init__(self, K, rseed=1):
        self.K = K
        self.dataset = Iris()
        self.rseed = rseed

    def task(self, name, x, y, split_idx=None):
        return Task('vector_classification',
                    name=name,
                    x=np.asarray(x),
                    y=np.asarray(y),
                    n_classes=3,
                    split_idx=split_idx,
                   )

    def protocol(self, algo, stop_after=None):
        x_all = np.asarray([
                [
                    m['sepal_length'],
                    m['sepal_width'],
                    m['petal_length'],
                    m['petal_width'],
                    ]
                for m in self.dataset.meta])
        y_all = np.asarray(int_labels([m['name'] for m in self.dataset.meta]))

        kf = cross_validation.KFold(len(y_all), self.K)
        idxmap = np.random.RandomState(self.rseed).permutation(len(y_all))

        losses = []

        for i, (train_idxs, test_idxs) in enumerate(kf):
            if stop_after is not None and i >= stop_after:
                break
            train = self.task(
                'train_%i' % i,
                x=x_all[idxmap[train_idxs]],
                y=y_all[idxmap[train_idxs]])
            test = self.task(
                'test_%i' % i,
                x=x_all[idxmap[test_idxs]],
                y=y_all[idxmap[test_idxs]])

            model = algo.best_model(train=train, valid=None)
            losses.append(algo.loss(model, test))

        return np.mean(losses)


class SimpleCrossValidation(object):
    """ Simple demo version of KfoldClassification that stops
    after a single fold for brevity.
    """
    def __init__(self):
        self.kfold = KfoldClassification(5)

    def protocol(self, algo):
        return self.kfold.protocol(algo, stop_after=1)


