"""
CIFAR-10 Image classification dataset

Data available from and described at:
http://www.cs.toronto.edu/~kriz/cifar.html

If you use this dataset, please cite "Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""

# Authors: James Bergstra <bergstra@rowland.harvard.edu>
# License: BSD 3 clause

import os
import pickle
import logging
import shutil

import numpy as np

from ..data_home import get_data_home
from ..utils.download_and_extract import download_and_extract

logger = logging.getLogger(__name__)

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']

class CIFAR10(object):
    """

    meta[i] is dict with keys:
        id: int identifier of this example
        label: int in range(10)
        split: 'train' or 'test'

    meta_const is dict with keys:
        image:
            shape: 32, 32, 3
            dtype: 'uint8'


    """

    DOWNLOAD_IF_MISSING = True  # the value when accessing .meta

    def __init__(self):
        self.meta_const = dict(
                image = dict(
                    shape = (32, 32, 3),
                    dtype = 'uint8',
                    )
                )
        self.descr = dict(
                n_classes = 10,
                )

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=self.DOWNLOAD_IF_MISSING)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    def home(self, *names):
        return os.path.join(get_data_home(), 'cifar10', *names)

    def fetch(self, download_if_missing, verbose=True):
        if os.path.isdir(self.home('cifar-10-batches-py')):
            return

        if not os.path.isdir(self.home()):
            if download_if_missing:
                os.makedirs(self.home())
            else:
                raise IOError(self.home())

        download_and_extract(URL, self.home(), verbose=verbose)

    def clean_up(self):
        logger.info('recursively erasing %s' % self.home())
        if os.path.isdir(self.home()):
            shutil.rmtree(self.home())

    def build_meta(self):
        try:
            self._pixels
        except AttributeError:
            # load data into class attributes _pixels and _labels
            pixels = np.zeros((60000, 32, 32, 3), dtype='uint8')
            labels = np.zeros(60000, dtype='int32')
            fnames = ['data_batch_%i'%i for i in range(1,6)]
            fnames.append('test_batch')

            # load train and validation data
            n_loaded = 0
            for i, fname in enumerate(fnames):
                data = self.unpickle(fname)
                assert data['data'].dtype == np.uint8
                def futz(X):
                    return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
                pixels[n_loaded:n_loaded + 10000] = futz(data['data'])
                labels[n_loaded:n_loaded + 10000] = data['labels']
                n_loaded += 10000
            assert n_loaded == len(labels)
            CIFAR10._pixels = pixels
            CIFAR10._labels = labels

            # -- mark memory as read-only to prevent accidental modification
            pixels.flags['WRITEABLE'] = False
            labels.flags['WRITEABLE'] = False

            assert LABELS == self.unpickle('batches.meta')['label_names']
        meta = [dict(
                    id=i,
                    split='train' if i < 50000 else 'test',
                    label=LABELS[l])
                for i,l in enumerate(self._labels)]
        return meta

    def unpickle(self, basename):
        fname = self.home('cifar-10-batches-py', basename)
        logger.info('loading file %s' % fname)
        with open(fname, 'rb') as fo:
            try:
                return pickle.load(fo, encoding='latin1')
            except:
                return pickle.load(fo)
