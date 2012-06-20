"""
Experiment views on the Iris data set.

"""

import numpy as np
from sklearn import cross_validation

from .dataset import Iris
from ..base import Task, Split
from ..utils import int_labels
from ..dslang import Average, Score, BestModel


class KfoldClassification(object):
    """
    Access train/test splits for K-fold cross-validation as follows:

    >>> self.splits[k].train.x
    >>> self.splits[k].train.y
    >>> self.splits[k].test.x
    >>> self.splits[k].test.y

    """

    def __init__(self, K, rseed=1, y_as_int=False):
        self.K = K
        self.dataset = Iris()
        self.x = np.asarray([
                [
                    m['sepal_length'],
                    m['sepal_width'],
                    m['petal_length'],
                    m['petal_width'],
                    ]
                for m in self.dataset.meta])
        if y_as_int:
            self.y = np.asarray(int_labels([m['name']
                for m in self.dataset.meta]))
        else:
            self.y = np.asarray([m['name']
                for m in self.dataset.meta])
        kf = cross_validation.KFold(len(self.y), K)
        idxmap = np.random.RandomState(rseed).permutation(len(self.y))

        self.splits = []
        for train_idxs, test_idxs in kf:
            self.splits.append(Split(
                    train=Task('vector_classification',
                        x=self.x[idxmap[train_idxs]],
                        y=self.y[idxmap[train_idxs]]),
                    test=Task('vector_classification',
                        x=self.x[idxmap[test_idxs]],
                        y=self.y[idxmap[test_idxs]]),
                    ))


        self.dsl = Average([Score(BestModel(s.train), s.test)
            for s in self.splits])

