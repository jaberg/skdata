import csv
import os

import numpy as np

import utils

class BuildOnInit(object):
    """
    """
    def __init__(self):
        try:
            self.meta
        except AttributeError:
            self.meta = self.build_meta()

    def cache(self):
        # cause future __init__ not to build_meta()
        self.__class__.meta = self.meta


class Iris(BuildOnInit):
    """Load and return the iris dataset (classification).

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


class Digits(BuildOnInit):
    """
    """
    def build_meta(self):
        module_path = os.path.dirname(__file__)
        data = np.loadtxt(os.path.join(module_path, 'data', 'digits.csv.gz'),
                          delimiter=',')
        descr = open(os.path.join(module_path, 'descr', 'digits.rst')).read()
        print data
        print descr
        return []
        target = data[:, -1]
        flat_data = data[:, :-1]
        images = flat_data.view()
        images.shape = (-1, 8, 8)

        if n_class < 10:
            idx = target < n_class
            flat_data, target = flat_data[idx], target[idx]
            images = images[idx]

        return Bunch(data=flat_data,
                     target=target.astype(np.int),
                     target_names=np.arange(10),
                     images=images,
                     DESCR=descr)

