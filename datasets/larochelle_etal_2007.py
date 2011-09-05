""" Functions related to the datasets used in Larochelle et al. 2007

Dataset:

- convex
- rectangles basic
- rectangles images
- mnist basic
- mnist rotated
- mnist background images
- mnist background random

These datasets were introduced in "An Empirical Evaluation of Deep Architectures
on Problems with Many Factors of Variation" Hugo Larochelle, Dumitru Erhan,
Aaron Courville, James Bergstra and Yoshua Bengio. In Proc of International
Conference on Machine Learning (2007).

url: http://www.iro.umontreal.ca/~lisa/twiki/pub/Public/
    DeepVsShallowComparisonICML2007/icml-2007-camera-ready.pdf

"""

# Authors: James Bergstra <bergstra@rowland.harvard.edu>
# License: BSD 3 clause

#
# ISSUES (XXX)
# ------------
# - These datasets are all built algorithmically by modifying image datasets.
#   It would be nice to have the code that did those modifications in this file.
#   The original matlab code is available here:
#     http://www.iro.umontreal.ca/~lisa/twiki/pub/Public/
#     DeepVsShallowComparisonICML2007/scripts_only.tar.gz
#

import array  # XXX why is this used not numpy?
import os
import sys

import numpy as np

from data_home import get_data_home
import utils

import logging
logger = logging.getLogger(__name__)


class AMat:
    """DataSource to access a plearn amat file as a periodic unrandomized stream.

    Attributes:

    input -- all columns of input
    target -- all columns of target
    weight -- all columns of weight
    extra -- all columns of extra

    all -- the entire data contents of the amat file
    n_examples -- the number of training examples in the file

    AMat stands for Ascii Matri[x,ces]

    """

    marker_size = '#size:'
    marker_sizes = '#sizes:'
    marker_col_names = '#:'

    def __init__(self, path, head=None):

        """Load the amat at <path> into memory.
        path - str: location of amat file
        head - int: stop reading after this many data rows

        """
        logger.info('Loading AMat: %s' % path)
        self.all = None
        self.input = None
        self.target = None
        self.weight = None
        self.extra = None

        self.header = False
        self.header_size = None
        self.header_rows = None
        self.header_cols = None
        self.header_sizes = None
        self.header_col_names = []

        data_started = False
        data = array.array('d')

        f = open(path)
        n_data_lines = 0
        len_float_line = None

        for i,line in enumerate(f):
            if n_data_lines == head:
                #we've read enough data, 
                # break even if there's more in the file
                break
            if len(line) == 0 or line == '\n':
                continue
            if line[0] == '#':
                if not data_started:
                    #the condition means that the file has a header, and we're on 
                    # some header line
                    self.header = True
                    if line.startswith(AMat.marker_size):
                        info = line[len(AMat.marker_size):]
                        self.header_size = [int(s) for s in info.split()]
                        self.header_rows, self.header_cols = self.header_size
                    if line.startswith(AMat.marker_col_names):
                        info = line[len(AMat.marker_col_names):]
                        self.header_col_names = info.split()
                    elif line.startswith(AMat.marker_sizes):
                        info = line[len(AMat.marker_sizes):]
                        self.header_sizes = [int(s) for s in info.split()]
            else:
                #the first non-commented line tells us that the header is done
                data_started = True
                float_line = [float(s) for s in line.split()]
                if len_float_line is None:
                    len_float_line = len(float_line)
                    if (self.header_cols is not None) \
                            and self.header_cols != len_float_line:
                        print >> sys.stderr, \
                                'WARNING: header declared %i cols but first line has %i, using %i',\
                                self.header_cols, len_float_line, len_float_line
                else:
                    if len_float_line != len(float_line):
                        raise IOError('wrong line length', i, line)
                data.extend(float_line)
                n_data_lines += 1

        f.close()

        # convert from array.array to np.ndarray
        nshape = (len(data) / len_float_line, len_float_line)
        self.all = np.frombuffer(data).reshape(nshape)
        self.n_examples = self.all.shape[0]
        logger.info('AMat loaded all shape: %s' % repr(self.all.shape))

        # assign
        if self.header_sizes is not None:
            if len(self.header_sizes) > 4:
                print >> sys.stderr, 'WARNING: ignoring sizes after 4th in %s' % path
            leftmost = 0
            #here we make use of the fact that if header_sizes has len < 4
            # the loop will exit before 4 iterations
            attrlist = ['input', 'target', 'weight', 'extra']
            for attr, ncols in zip(attrlist, self.header_sizes): 
                setattr(self, attr, self.all[:, leftmost:leftmost+ncols])
                leftmost += ncols
            logger.info('AMat loaded %s shape: %s' % (attr,
                repr(getattr(self, attr).shape)))
        else:
            logger.info('AMat had no header: %s' % path)


class BaseL2007(object):
    """Base class for fetching and loading Larochelle etal datasets

    This class has functionality to:

    - download the dataset from the internet  (in amat format)
    - convert the dataset from amat format to npy format
    - load the dataset from either amat or npy source files

    """

    BASE_URL = 'http://www.iro.umontreal.ca/~lisa/icml2007data'
    DOWNLOAD_IF_MISSING = True  # value used on first access to .meta
    MMAP_MODE = 'r'             # _labels and _inputs are loaded this way.
                                # See numpy.load / numpy.memmap for semantics.

    meta_const = dict(
            image=dict(
                shape=(28, 28),
                dtype='float32'))

    def home(self, *names):
        return os.path.join(
                get_data_home(),
                'larochelle_etal_2007',
                *names)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self, download_if_missing):
        try:
            open(self.home(self.NAME+'_inputs.npy')).close()
            open(self.home(self.NAME+'_labels.npy')).close()
        except IOError:
            if download_if_missing:
                try:
                    amat_test = AMat(self.home(self.AMAT + '_test.amat'))
                except IOError:
                    logger.info('Failed to read %s, downloading %s' %(
                        self.home(self.AMAT + '_test.amat'),
                        os.path.join(self.BASE_URL, self.REMOTE)))
                    if not os.path.exists(self.home()):
                        os.makedirs(self.home())
                    utils.download_and_extract(
                        os.path.join(self.BASE_URL, self.REMOTE),
                        self.home(),
                        verbose=False)
                    amat_test = AMat(self.home(self.AMAT + '_test.amat'))
                amat_train = AMat(self.home(self.AMAT + '_train.amat'))
                n_inputs = 28**2
                n_train = self.descr['n_train']
                n_valid = self.descr['n_valid']
                n_test = self.descr['n_test']
                assert amat_train.all.shape[0] == n_train + n_valid
                assert amat_test.all.shape[0] == n_test
                assert amat_train.all.shape[1] == amat_test.all.shape[1]
                assert amat_train.all.shape[1] == n_inputs + 1
                allmat = np.vstack((amat_train.all, amat_test.all))
                inputs = np.reshape(
                        allmat[:, :n_inputs].astype('float32'),
                        (-1, 28, 28))
                labels = allmat[:, n_inputs].astype('int32')
                assert np.all(labels == allmat[:, n_inputs])
                assert np.all(labels < self.descr['n_classes'])
                np.save(self.home(self.NAME + '_inputs.npy'), inputs)
                np.save(self.home(self.NAME + '_labels.npy'), labels)
            else:
                raise

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=self.DOWNLOAD_IF_MISSING)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    def build_meta(self):
        try:
            self._labels
        except AttributeError:
            # load data into class attributes _pixels and _labels
            inputs = np.load(self.home(self.NAME + '_inputs.npy'),
                    mmap_mode=self.MMAP_MODE)
            labels = np.load(self.home(self.NAME + '_labels.npy'),
                    mmap_mode=self.MMAP_MODE)
            self.__class__._inputs = inputs
            self.__class__._labels = labels
            assert len(inputs) == len(labels)

        def split_of_pos(i):
            if i < self.descr['n_train']:
                return 'train'
            if i < self.descr['n_train'] + self.descr['n_valid']:
                return 'valid'
            return 'test'

        assert len(self._labels) == sum(
                [self.descr[s] for s in 'n_train', 'n_valid', 'n_test'])

        meta = [dict(id=i, split=split_of_pos(i), label=l)
                for i,l in enumerate(self._labels)]

        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Driver routines to be called by datasets.main
    # ------------------------------------------------------------------------

    @classmethod
    def main_fetch(cls):
        cls().fetch(download_if_missing=True)

    @classmethod
    def main_show(cls):
        from utils.glviewer import glumpy_viewer, glumpy
        self = cls()
        labels = [m['label'] for m in self.meta]
        glumpy_viewer(
                img_array=self._inputs,
                arrays_to_print=[labels],
                cmap=glumpy.colormap.Grey,
                window_shape=(28 * 3, 28 * 3))

    @classmethod
    def main_clean_up(cls):
        cls().clean_up()


#
# MNIST Variations
#

class BaseMNIST(BaseL2007):
    descr = dict(
            n_classes=10,
            n_train=10000,
            n_valid=2000,
            n_test=50000
            )


class MNIST_Basic(BaseMNIST):
    REMOTE = 'mnist.zip'  # fetch BASE_URL/REMOTE
    REMOTE_SIZE = '23M'
    AMAT = 'mnist'        # matches name unzip'd from REMOTE
    NAME = 'mnist_basic'  # use this as root filename for saved npy files


class MNIST_BackgroundImages(BaseMNIST):
    REMOTE = 'mnist_background_images.zip'
    REMOTE_SIZE = '88M'
    AMAT = 'mnist_background_images'
    NAME = 'mnist_background_images'


class MNIST_BackgroundRandom(BaseMNIST):
    REMOTE = 'mnist_background_random.zip',
    REMOTE_SIZE = '219M'
    AMAT = 'mnist_background_random'
    NAME = 'mnist_background_random'


class MNIST_Rotated(BaseMNIST):
    """
    There are two versions of this dataset available for download.

    1. the original dataset used in the ICML paper.

    2. a corrected dataset used to produce revised numbers for the web page
       version of the paper.

    This class loads the corrected dataset.
    """
    REMOTE = 'mnist_rotation_new.zip'
    REMOTE_SIZE = '56M'
    NAME = 'mnist_rotated'

    if 0:
        self.amat_filename_test = os.path.join(rootdir,
            'mnist_all_rotation_normalized_float_test.amat')
        self.amat_filename_train=os.path.join(rootdir,
            'mnist_all_rotation_normalized_float_train_valid.amat'),
        self.npy_filename_root=os.path.join(rootdir, 'mnist_rotated'),


class MNIST_RotatedBackgroundImages(BaseMNIST):
    """
    There are two versions of this dataset available for download.

    1. the original dataset used in the ICML paper.

    2. a corrected dataset used to produce revised numbers for the web page
       version of the paper.

    This class loads the corrected dataset.
    """
    REMOTE = 'mnist_rotation_back_image_new.zip'
    REMOTE_SIZE = '115M'
    NAME = 'mnist_rotated_background_images'
    if 0:
        amat_filename_test=os.path.join(
                rootdir,
                'mnist_all_background_images_rotation_normalized_test.amat')
        amat_filename_train=os.path.join(
                rootdir,
                'mnist_all_background_images_rotation_normalized_train_valid.amat')
        npy_filename_root=os.path.join(
                rootdir,
                'mnist_rotated_background_images')


class MNIST_Noise(BaseMNIST):
    def __init__(self, level):
        if level not in range(1, 7):
            raise ValueError('Noise level must be an int 1 <= level <= 6', level)
        self.level = level
        self.REMOTE = 

        self.http_source='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_noise_variation.tar.gz',
        self.amat_filename_all=os.path.join(rootdir,
            'mnist_noise_variations_all_%i.amat'%level),
        self.npy_filename_root=os.path.join(rootdir, 'mnist_noise_%i'%level),


#
# Rectangles Variations
#

class Rectangles(BaseL2007):
    REMOTE = 'rectangles.zip'
    REMOTE_SIZE = '2.7M'
    AMAT = 'rectangles'
    NAME = 'rectangles'
    descr = dict(
        n_classes=2,
        n_train=1000,
        n_valid=200,
        n_test=50000)


class RectanglesImages(BaseL2007):
    REMOTE = 'rectangles_images.zip'
    REMOTE_SIZE = '82M'
    AMAT = 'rectangles_im'
    NAME = 'rectangles_images'
    descr = dict(
        n_classes=2,
        n_train=10000,
        n_valid=2000,
        n_test=50000)


#
# Convex
#

class Convex(BaseL2007):
    REMOTE = 'convex.zip'
    REMOTE_SIZE = '3.4M'
    AMAT = 'convex'
    NAME = 'convex'
    descr = dict(
        n_classes=2,
        n_train=6500,
        n_valid=1500,
        n_test=50000)
