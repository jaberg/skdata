"""
Various routines to load/access faces datasets.
"""
from __future__ import absolute_import

import os
import numpy
import pylab as pl
from .config import data_root # config
from .dataset import Dataset

def att(path=None, randomize=True, normalize=True):
    path = os.path.join(data_root(), 'faces','att','orl_faces')\
           if path is None else path
    
    h, w = 112, 92
    nsubjects = 40
    npics = 10

    x = numpy.zeros((nsubjects * npics, h * w))
    y = numpy.zeros(nsubjects * npics)

    for sid in range(nsubjects):
        sdir = os.path.join(path, 's%i'%(sid+1))
        for n in range(npics):
            img = pl.imread(os.path.join(sdir,'%i.pgm'%(n+1)))
            x[sid*npics + n,:] = img[::-1,:].flatten()
            y[sid*npics + n] = sid

    if normalize:
        x *= (1.0 / 255.0)

    perm = numpy.random.permutation(len(x))

    rval = Dataset()
    rval.n_classes = nsubjects
    rval.img_shape = (112,92)
    rval.train = Dataset.Obj(x=x[perm,:], y=y[perm])

    # Not sure how well dataset lends itself to classification (only 400 images!)
    # therefore not too sure it makes sense to have a train/test split
    rval.valid = None
    rval.test = None

    return rval
