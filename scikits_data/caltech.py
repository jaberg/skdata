"""
Various routines to load/access MNIST data.
"""

import os
import numpy

from pylearn.io.pmat import PMat
from pylearn.datasets.config import data_root # config
from pylearn.datasets.dataset import Dataset

def caltech_silhouette():

    rval = Dataset()

    
    path = os.path.join(data_root(), 'caltech_silhouettes')

    rval.train = Dataset.Obj(x=numpy.load(os.path.join(path,'train_data.npy')),
                             y=numpy.load(os.path.join(path,'train_labels.npy')))
    rval.valid = Dataset.Obj(x=numpy.load(os.path.join(path,'val_data.npy')),
                             y=numpy.load(os.path.join(path,'val_labels.npy')))
    rval.test  = Dataset.Obj(x=numpy.load(os.path.join(path,'test_data.npy')),
                             y=numpy.load(os.path.join(path,'test_labels.npy')))

    rval.n_classes = 101
    rval.img_shape = (28,28)

    return rval
