"""
Various routines to load/access MNIST data.
"""
from __future__ import absolute_import

import os
import numpy
import cPickle

import logging
_logger = logging.getLogger('pylearn.datasets.cifar10')

from pylearn.datasets.config import data_root # config
from pylearn.datasets.dataset import Dataset # dataset.py

def unpickle(file):
    fname = os.path.join(data_root(),
            'cifar10', 
            'cifar-10-batches-py',
            file)
    _logger.info('loading file %s' % fname)
    fo = open(fname, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class cifar10(object):
    """

    This class gives access to meta-data of cifar10 dataset.
    The constructor loads it from <data>/cifar10/cifar-10-batches-py/
    where <data> is the pylearn data root (os.getenv('PYLEARN_DATA_ROOT')).

    Attributes:

    self.img_shape - the unrasterized image shape of each row in all.x
    self.img_size - the number of pixels in (aka length of) each row
    self.n_classes - the number of labels in the dataset (10)

    self.all.x    matrix - all train and test images as rasterized rows
    self.all.y    vector - all train and test labels as integers
    self.train.x  matrix - first ntrain rows of all.x
    self.train.y  matrix - first ntrain elements of all.y
    self.valid.x  matrix - rows ntrain to ntrain+nvalid of all.x
    self.valid.y  vector - elements ntrain to ntrain+nvalid of all.y
    self.test.x   matrix - rows ntrain+valid to end of all.x
    self.test.y   vector - elements ntrain+valid to end of all.y

    """

    def __init__(self, dtype='uint8', ntrain=40000, nvalid=10000, ntest=10000):
        assert ntrain + nvalid <= 50000
        assert ntest <= 10000

        self.img_shape = (3,32,32)
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 10

        lenx = numpy.ceil((ntrain + nvalid) / 10000.)*10000
        x = numpy.zeros((lenx,self.img_size), dtype=dtype)
        y = numpy.zeros(lenx, dtype=dtype)
        
        fnames = ['data_batch_%i'%i for i in range(1,6)]

        # load train and validation data
        nloaded = 0
        for i, fname in enumerate(fnames):
            data = unpickle(fname)
            x[i*10000:(i+1)*10000, :] = data['data']
            y[i*10000:(i+1)*10000] = data['labels']

            nloaded += 10000
            if nloaded >= ntrain + nvalid + ntest: break;

        self.all = Dataset.Obj(x=x, y=y)
        
        self.train = Dataset.Obj(x=x[0:ntrain], y=y[0:ntrain])
        self.valid = Dataset.Obj(x=x[ntrain:ntrain+nvalid],
                                 y=y[ntrain:ntrain+nvalid])
       
        # load test data
        data = unpickle('test_batch')
        self.test = Dataset.Obj(x=data['data'][0:ntest],
                                y=data['labels'][0:ntest])

    def preprocess(self, x):
        return numpy.float64( x *1.0 / 255.0)

def first_1k(dtype='uint8', ntrain=1000, nvalid=200, ntest=200):
    return cifar10(dtype, ntrain, nvalid, ntest)

def tile_rasterized_examples(X, img_shape=(32,32)):
    """Returns an ndarray that is ready to be passed to `image_tiling.save_tiled_raster_images`

    This function is for the `x` matrices in the cifar dataset, or for the weight matrices
    (filters) used to multiply them.
    """
    ndim = img_shape[0]*img_shape[1]
    assert ndim *3 == X.shape[1], (ndim, X.shape)
    X = X.astype('float32')
    r = X[:,:ndim]
    g = X[:,ndim:ndim*2]
    b = X[:,ndim*2:]
    from pylearn.io.image_tiling import tile_raster_images
    rval = tile_raster_images((r,g,b,None), img_shape=img_shape)
    return rval


