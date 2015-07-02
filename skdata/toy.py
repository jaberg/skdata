"""
Several small non-synthetic datasets that do not require any downloading.

"""
import csv
import os

import numpy as np

from utils import image


class BuildOnInit(object):
    """Base class that calls build_meta and build_all
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


class Linnerud(BuildOnInit):
    """Dataset of exercise and physiological measurements (regression).

    meta[i] is dict of
        weight: float
        waist: float
        pulse: float
        chins: float
        situps: float
        jumps: float
    """
    def build_all(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'data/')
        data_exercise = np.loadtxt(base_dir + 'linnerud_exercise.csv', skiprows=1)
        data_physiological = np.loadtxt(base_dir + 'linnerud_physiological.csv',
                                        skiprows=1)
        #header_physiological == ['Weight', 'Waist', 'Pulse']
        #header_exercise == ['Chins', 'Situps', 'Jumps']
        assert data_exercise.shape == (20, 3)
        assert data_physiological.shape == (20, 3)
        meta = [dict(weight=p[0], waist=p[1], pulse=p[2],
                     chins=e[0], situps=e[1], jumps=e[2])
                for e, p in zip(data_exercise, data_physiological)]
        descr = open(os.path.dirname(__file__) + '/descr/linnerud.rst').read()
        return meta, dict(txt=descr), {}

    def regression_task(self):
        # Task as defined on pg 15 of
        #    Tenenhaus, M. (1998). La regression PLS: theorie et pratique.
        #    Paris: Editions Technic.
        X = [(m['weight'], m['waist'], m['pulse']) for m in self.meta]
        Y = [(m['chins'], m['situps'], m['jumps']) for m in self.meta]
        return np.asarray(X, dtype=np.float), np.asarray(Y, dtype=np.float)


class Boston(BuildOnInit):
    """Dataset of real estate features (regression)

    meta[i] is dict of
        CRIM: float
        ZN: float
        INDUS: float
        CHAS: float
        NOX: float
        RM: float
        AGE: float
        DIS: float
        RAD: float
        TAX: float
        PTRATIO: float
        B: float
        LSTAT: float
        MEDV: float

    descr is dict of
        txt: textual description of dataset
        X_features: list of keys for standard regression task
        Y_features: list of keys for standard regression task

    The standard regression task is to predict MEDV (median value?) from the
    other features.
    """
    def build_all(self):
        module_path = os.path.dirname(__file__)
        descr = open(os.path.join(
            module_path, 'descr', 'boston_house_prices.rst')).read()
        data_file = csv.reader(open(os.path.join(
            module_path, 'data', 'boston_house_prices.csv')))
        n_samples, n_features = [int(t) for t in data_file.next()[:2]]
        feature_names = data_file.next()
        meta = [dict(zip(feature_names, map(float, d))) for d in data_file]
        return (meta,
                dict(txt=descr,
                    X_features=feature_names[:-1],
                    Y_features=feature_names[-1:]),
                {})

    def regression_task(self):
        X_features = self.descr['X_features']
        Y_features = self.descr['Y_features']
        X = map(lambda m: [m[f] for f in X_features], self.meta)
        Y = map(lambda m: [m[f] for f in Y_features], self.meta)
        return np.asarray(X, np.float), np.asarray(Y, np.float)


class SampleImages(BuildOnInit):
    """Dataset of 2 sample jpg images (no specific task)

    meta[i] is dict of:
        filename: str (relative to self.imgdir)
    """
    def __init__(self):
        self.imgdir = os.path.join(os.path.dirname(__file__), "images")
        BuildOnInit.__init__(self)

    def fullpath(self, relpath):
        return os.path.join(self.imgdir, relpath)

    def build_all(self):
        descr = open(os.path.join(self.imgdir, 'README.txt')).read()
        meta = [dict(filename=filename)
                     for filename in os.listdir(self.imgdir)
                     if filename.endswith(".jpg")]
        return (meta,
                dict(txt=descr),
                {})

    def images(self):
        return map(image.load_rgb_f32,
                   map(lambda m: self.fullpath(m['filename']), self.meta))
