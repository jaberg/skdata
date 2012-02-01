"""
Diabetes - a small non-synthetic dataset for binary classification with
temporal data.

http://archive.ics.uci.edu/ml/datasets/Diabetes

"""
import csv
import os

import numpy as np

import utils

from .toy import BuildOnInit

class Diabetes(BuildOnInit):
    """Dataset of diabetes results (classification)

    meta[i] is dict with
        data: ?
        label: ?

    """
    # XXX:  what is this data?
    def build_meta(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'data')
        data = np.loadtxt(os.path.join(base_dir, 'diabetes_data.csv.gz'))
        target = np.loadtxt(os.path.join(base_dir, 'diabetes_target.csv.gz'))
        itarget = map(int, target)
        assert all(itarget == target)
        assert len(data) == len(target)
        return [dict(d=d, l=l) for (d,l) in zip(data, itarget)]

    def classification_task(self):
        X = np.asarray([m['d'] for m in self.meta])
        y = np.asarray([m['l'] for m in self.meta])
        return X, y


