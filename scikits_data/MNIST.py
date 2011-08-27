"""
Various routines to load/access MNIST data.
"""

import os
import numpy

from pylearn.io.pmat import PMat
from pylearn.datasets.config import data_root # config
from pylearn.datasets.dataset import Dataset

def head(n=10, path=None):
    """Load the first MNIST examples.

    Returns two matrices: x, y.  x has N rows of 784 columns.  Each row of x represents the
    28x28 grey-scale pixels in raster order.  y is a vector of N integers.  Each element y[i]
    is the label of the i'th row of x.

    """
    if path is None:
      path = os.path.join(data_root(), 'mnist','mnist_all.pmat')

    dat = PMat(fname=path)

    rows=dat.getRows(0,n)

    return rows[:,0:-1], numpy.asarray(rows[:,-1], dtype='int8')


#What is the purpose of this fct?
#If still usefull, rename it as it conflict with the python an numpy nake all.
#def all(path=None):
#    return head(n=None, path=path)

def train_valid_test(ntrain=50000, nvalid=10000, ntest=10000, path=None):
    all_x, all_targ = head(ntrain+nvalid+ntest, path=path)

    rval = Dataset()

    rval.train = Dataset.Obj(x=all_x[0:ntrain],
            y=all_targ[0:ntrain])
    rval.valid = Dataset.Obj(x=all_x[ntrain:ntrain+nvalid],
            y=all_targ[ntrain:ntrain+nvalid])
    rval.test =  Dataset.Obj(x=all_x[ntrain+nvalid:ntrain+nvalid+ntest],
            y=all_targ[ntrain+nvalid:ntrain+nvalid+ntest])

    rval.n_classes = 10
    rval.img_shape = (28,28)
    return rval


def full():
    return train_valid_test()

#useful for test, keep it
def first_10():
    return train_valid_test(ntrain=10, nvalid=10, ntest=10)

#useful for test, keep it
def first_100():
    return train_valid_test(ntrain=100, nvalid=100, ntest=100)

def first_1k():
    return train_valid_test(ntrain=1000, nvalid=200, ntest=200)

def first_10k():
    return train_valid_test(ntrain=10000, nvalid=2000, ntest=2000)

