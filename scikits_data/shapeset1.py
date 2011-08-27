"""
Routines to load/access Shapeset1
"""

import os
import numpy

from pylearn.io.amat import AMat
from pylearn.datasets.config import data_root
from pylearn.datasets.dataset import Dataset

def _head(path, n):
    dat = AMat(path=path, head=n)

    try:
        assert dat.input.shape[0] == n
        assert dat.target.shape[0] == n
    except Exception , e:
        raise Exception("failed to read %i lines from file %s" % (n, path))

    return dat.input, numpy.asarray(dat.target, dtype='int64').reshape(dat.target.shape[0])


def head_train(n=10000):
    """Load the first Shapeset1 training examples.

    Returns two matrices: x, y.
    x has N rows of 1024 columns.
    Each row of x represents the 32x32 grey-scale pixels in raster order.
    y is a vector of N integers between 0 and 2.
    Each element y[i] is the label of the i'th row of x.
    """
    path = os.path.join(data_root(), 'shapeset1','shapeset1_1cspo_2_3.10000.train.shape.amat')
    return _head(path, n)

def head_valid(n=5000):
    """Load the first Shapeset1 validation examples.

    Returns two matrices: x, y.
    x has N rows of 1024 columns.
    Each row of x represents the 32x32 grey-scale pixels in raster order.
    y is a vector of N integers between 0 and 2.
    Each element y[i] is the label of the i'th row of x.
    """
    path = os.path.join(data_root(), 'shapeset1','shapeset1_1cspo_2_3.5000.valid.shape.amat')
    return _head(path, n)

def head_test(n=5000):
    """Load the first Shapeset1 testing examples.

    Returns two matrices: x, y.
    x has N rows of 1024 columns.
    Each row of x represents the 32x32 grey-scale pixels in raster order.
    y is a vector of N integers between 0 and 2.
    Each element y[i] is the label of the i'th row of x.
    """
    path = os.path.join(data_root(), 'shapeset1','shapeset1_1cspo_2_3.5000.test.shape.amat')
    return _head(path, n)

def train_valid_test(ntrain=10000, nvalid=5000, ntest=5000):
    train_x, train_y = head_train(n=ntrain)
    valid_x, valid_y = head_valid(n=nvalid)
    test_x,  test_y  = head_test(n=ntest)

    rval = Dataset()
    rval.train = Dataset.Obj(x = train_x, y = train_y)
    rval.valid = Dataset.Obj(x = valid_x, y = valid_y)
    rval.test  = Dataset.Obj(x = test_x,  y = test_y)

    rval.n_classes = 3
    rval.img_shape = (32, 32)

    return rval


