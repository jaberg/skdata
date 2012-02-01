"""
Digits - small non-synthetic dataset of 10-way image classification

XXX: What's the source on this dataset?

"""
import csv
import os

import numpy as np

from .toy import BuildOnInit

class Digits(BuildOnInit):
    """Dataset of small digit images (classification)

    meta[i] is dict with
        img: an 8x8 ndarray
        label: int 0 <= label < 10
    """
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

    # XXX: is img JSON-encodable ?
    # return img_classification_task interface

