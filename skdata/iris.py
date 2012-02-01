"""
Iris - a small non-synthetic dataset not requiring download.

"""
import csv
import os

import numpy as np

import utils

from .toy import BuildOnInit

class Iris(BuildOnInit):
    """Dataset of flower properties (classification)

    self.meta has elements with following structure:

        meta[i] = dict
            sepal_length: float
            sepal_width: float
            petal_length: float
            petal_width: float
            name: one of 'setosa', 'versicolor', 'virginica'
    """
    def build_meta(self):
        module_path = os.path.dirname(__file__)
        data_file = csv.reader(open(os.path.join(
            module_path, 'data', 'iris.csv')))
        fdescr = open(os.path.join(module_path, 'descr', 'iris.rst'))
        temp = data_file.next()
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        temp = list(data_file)
        data = [map(float, t[:-1]) for t in temp]
        target = [target_names[int(t[-1])] for t in temp]
        meta = [dict(
            sepal_length=d[0],
            sepal_width=d[1],
            petal_length=d[2],
            petal_width=d[3],
            name=t)
                for d, t in zip(data, target)]
        return meta


    def classification_task(self):
        X = [[m['sepal_length'], m['sepal_width'],
            m['petal_length'], m['petal_width']]
                for m in self.meta]
        y = utils.int_labels([m['name'] for m in self.meta])
        return np.asarray(X), np.asarray(y)


