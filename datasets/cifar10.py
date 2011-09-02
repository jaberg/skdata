"""
CIFAR-10 Image classification dataset

Data available from and described at:
http://www.cs.toronto.edu/~kriz/cifar.html

If you use this dataset, please cite "Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""
import os
import logging
import urllib
import shutil

import numpy as np

from data_home import get_data_home
import utils.download_and_extract

logger = logging.getLogger(__name__)

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

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

class CIFAR10(object):
    """

    """

    def home(self, *names):
        return os.path.join(get_data_home(), 'cifar10', *names)

    def fetch(self, download_if_missing):
        if os.path.isdir(self.home('cifar-10-batches-py')):
            return

        if not os.path.isdir(self.home()):
            if download_if_missing:
                os.makedirs(self.home())
            else:
                raise IOError(self.home())

        utils.download_and_extract.download_and_extract(
                URL, self.home(), overwrite=True)

    def __init__(self, dtype='uint8', ntrain=40000, nvalid=10000, ntest=10000):
        if 0:
            assert ntrain + nvalid <= 50000
            assert ntest <= 10000

            self.img_shape = (3,32,32)
            self.img_size = np.prod(self.img_shape)
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

def main_fetch():
    CIFAR10().fetch(True)
