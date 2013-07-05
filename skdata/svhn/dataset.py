# -*- coding: utf-8 -*-
"""The Street View House Numbers (SVHN) Dataset

SVHN is a real-world image dataset for developing machine learning and
object recognition algorithms with minimal requirement on data
preprocessing and formatting. It can be seen as similar in flavor to
MNIST (e.g., the images are of small cropped digits), but incorporates
an order of magnitude more labeled data (over 600,000 digit images) and
comes from a significantly harder, unsolved, real world problem
(recognizing digits and numbers in natural scene images). SVHN is
obtained from house numbers in Google Street View images. 

Overview
--------

    * 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label
    9 and '0' has label 10.

    * 73257 digits for training, 26032 digits for testing, and 531131
    additional, somewhat less difficult samples, to use as extra
    training data

    * Comes in two formats:
        1. Original images with character level bounding boxes.
        2. MNIST-like 32-by-32 images centered around a single character
        (many of the images do contain some distractors at the sides).

Reference
---------

Please cite the following reference in papers using this dataset: Yuval
Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng
Reading Digits in Natural Images with Unsupervised Feature Learning NIPS
Workshop on Deep Learning and Unsupervised Feature Learning 2011.

http://ufldl.stanford.edu/housenumbers

For questions regarding the dataset, please contact
streetviewhousenumbers@gmail.com

"""

# Copyright (C) 2012
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          James Bergstra

# License: Simplified BSD

import logging
import os
from os import path
import shutil

import lockfile

from skdata.data_home import get_data_home
from skdata.utils import download

log = logging.getLogger(__name__)
BASE_URL = "http://ufldl.stanford.edu/housenumbers/"


class CroppedDigits(object):
    """XXX

    Notes
    -----
    If joblib is available, then `meta` will be cached for faster
    processing. To install joblib use 'pip install -U joblib' or
    'easy_install -U joblib'.
    """

    FILES = dict(
        train=('train_32x32.mat', 'e6588cae42a1a5ab5efe608cc5cd3fb9aaffd674'),
        test=('test_32x32.mat', '29b312382ca6b9fba48d41a7b5c19ad9a5462b20'),
        extra=('extra_32x32.mat', 'd7d93fbeec3a7cf69236a18015d56c7794ef7744'),
        )

    def __init__(self, need_extra=True):

        self.name = self.__class__.__name__
        self.need_extra=need_extra

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'), verbose=False)
            self._get_meta = mem.cache(self._get_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""

        home = self.home()

        if not download_if_missing:
            raise IOError("'%s' exists!" % home)

        lock = lockfile.FileLock(home)
        if lock.is_locked():
            log.warn('%s is locked, waiting for release' % home)

        with lock:
            for fkey, (fname, sha1) in self.FILES.iteritems():
                url = path.join(BASE_URL, fname)
                basename = path.basename(url)
                archive_filename = self.home(basename)
                marker = self.home(basename + '.marker')
                
                if ('extra' not in url) or self.need_extra:
                    if not path.exists(marker):
                        if not download_if_missing:
                            return
                        if not path.exists(home):
                            os.makedirs(home)
                        download(url, archive_filename, sha1=sha1)
                        open(marker, 'w').close()

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):
        meta = dict([(k, {'filename': self.home(v[0])})
                     for k, v in self.FILES.iteritems()])
        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())
