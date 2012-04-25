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
import cPickle
import logging
import shutil

import numpy as np

from data_home import get_data_home
import utils.download_and_extract

logger = logging.getLogger(__name__)

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']

# XXX: Consider modifying this file to use mem-mapped data
#      Modify fetch to unpickle the arrays, and write a mem-mappable numpy
#      array.  Consider memmapping the float32 version of the data rather than
#      the uint8, since that's the case where memory gets more important.

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

    def fetch(self, download_if_missing):
        if os.path.isdir(self.home('cifar-10-batches-py')):
            return

        if not os.path.isdir(self.home()):
            if download_if_missing:
                os.makedirs(self.home())
            else:
                raise IOError(self.home())

        utils.download_and_extract.download_and_extract(
                URL, self.home())

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
        fo = open(fname, 'rb')
        data = cPickle.load(fo)
        fo.close()
        return data

    def classification_task(self):
        #XXX: use .meta
        self.meta # triggers load if necessary
        y = self._labels
        X = self.latent_structure_task()
        return X, y

    def img_classification_task(self, dtype='uint8'):
        #XXX: use .meta
        self.meta # triggers load if necessary
        y = self._labels
        X = self._pixels.astype(dtype)
        if 'float' in dtype:
            X = X / 255.0
        return X, y

    def latent_structure_task(self):
        self.meta # triggers load if necessary
        return self._pixels.reshape((60000, 3072)).astype('float32') / 255


def main_fetch():
    CIFAR10().fetch(True)


def main_show():
    self = CIFAR10()
    from utils.glviewer import glumpy_viewer, glumpy
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=CIFAR10._pixels,
            arrays_to_print=[Y],
            cmap=glumpy.colormap.Grey,
            window_shape=(32 * 2, 32 * 2))


def main_clean_up():
    CIFAR10().clean_up()
