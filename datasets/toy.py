import csv
import os

import numpy as np

import utils

class BuildOnInit(object):
    """
    """
    def __init__(self):
        try:
            self.meta, self.descr, self.meta_const
        except AttributeError:
            meta, descr, meta_const = self.build_all()
            self.meta = meta
            self.descr = descr
            self.meta_const = meta_const


    def memoize(self):
        # cause future __init__ not to build_meta()
        self.__class__.meta = self.meta
        self.__class__.descr = self.descr
        self.__class__.meta_const = self.meta_const

    def build_all(self):
        return self.build_meta(), self.build_descr(), self.build_meta_const()

    def build_descr(self):
        return {}

    def build_meta_const(self):
        return {}


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
    meta[i] is dict with
        img: an 8x8 ndarray
        label: int 0 <= label < 10
    """
    # XXX: is img JSON-encodable ?
    def build_all(self):
        module_path = os.path.dirname(__file__)
        data = np.loadtxt(os.path.join(module_path, 'data', 'digits.csv.gz'),
                          delimiter=',')
        descr = open(os.path.join(module_path, 'descr', 'digits.rst')).read()
        target = data[:, -1]
        images = np.reshape(data[:, :-1], (-1, 8, 8))
        assert len(images) == len(target)
        itarget = map(int, target)
        assert all(itarget == target)
        meta = [dict(img=i, label=t) for i, t in zip(images, itarget)]
        return meta, descr, {}

    def classification_task(self):
        X = np.asarray([m['img'].flatten() for m in self.meta])
        y = np.asarray([m['label'] for m in self.meta])
        return X, y


class Diabetes(BuildOnInit):
    """

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

