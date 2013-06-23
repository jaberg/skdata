"""
Experiment views on the Iris data set.

"""

import numpy as np
from sklearn import cross_validation

from .dataset import RestaurantInspectionScores
from skdata.base import Task

def remove_dups(lst):
    rval = []
    seen = set()
    for l in lst:
        if l not in seen:
            rval.append(l)
            seen.add(l)
    return rval


class LocationClassification5(object):
    """
    Access train/test splits for K-fold cross-validation as follows:

    """

    def __init__(self, K, rseed=1):
        self.K = K
        self.dataset = RestaurantInspectionScores()
        self.rseed = rseed

    def task(self, name, x, y, split_idx=None):
        return Task('vector_classification',
                    name=name,
                    x=np.asarray(x),
                    y=np.asarray(y),
                    n_classes=3,
                    split_idx=split_idx,
                   )

    def protocol(self, algo):
        meta = list(self.dataset.meta)
        np.random.RandomState(self.rseed).shuffle(meta)
        indexable_names = np.asarray(
            remove_dups(m['restaurant_name'] for m in meta))
        kf = cross_validation.KFold(len(indexable_names), self.K)

        losses = []

        for i, (train_name_idxs, test_name_idxs) in enumerate(kf):
            train_names = set(indexable_names[train_name_idxs])
            test_names = set(indexable_names[test_name_idxs])

            # TODO: there is a numpy idiom for this right?
            #  (searchsorted?)
            def task_of_names(names):
                try:
                    x = [(m['address']['latitude'], m['address']['longitude'])
                        for m in meta if m['restaurant_name'] in names]
                except KeyError:
                    print m
                    raise
                y = [int((m['score'] - 50) / 10)
                        for m in meta if m['restaurant_name'] in names]
                return self.task( 'train_%i' % i, x=x, y=y)

            model = algo.best_model(
                train=task_of_names(train_names),
                valid=None)
            losses.append(
                algo.loss(model,
                    task=task_of_names(test_names)))

        return np.mean(losses)

