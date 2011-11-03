# -*- coding: utf-8 -*-
"""Caltech Object Datasets

Caltech 101: http://www.vision.caltech.edu/Image_Datasets/Caltech101
Caltech 256: http://www.vision.caltech.edu/Image_Datasets/Caltech256

If you make use of this data, please cite the following papers:
http://www.vision.caltech.edu/Image_Datasets/Caltech256/

The Caltech-256
Griffin, G. Holub, AD. Perona, P.
Caltech Technical Report (2007)
http://www.vision.caltech.edu/Image_Datasets/Caltech256/paper/256.pdf

Learning generative visual models from few training examples: an incremental
Bayesian approach tested on 101 object categories.
L. Fei-Fei, R. Fergus and P. Perona.
IEEE CVPR, Workshop on Generative-Model Based Vision (2004)
http://www.vision.caltech.edu/feifeili/Fei-Fei_GMBV04.pdf
"""

# Copyright (C) 2011
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD

# XXX: standard categorization tasks (csv-based)


import os
from os import path
import shutil
from glob import glob
import hashlib

import numpy as np

import larray
from data_home import get_data_home
from utils import download, extract, int_labels
from utils.image import ImgLoader


class BaseCaltech(object):
    """Caltech Object Dataset

    Attributes
    ----------
    meta: list of dict
        Metadata associated with the dataset. For each image with index i,
        meta[i] is a dict with keys:
            name: str
                Name of the individual's face in the image.
            filename: str
                Full path to the image.
            id: int
                Identifier of the image.
            sha1: str
                SHA-1 hash of the image.

    Notes
    -----
    If joblib is available, then `meta` will be cached for faster
    processing. To install joblib use 'pip install -U joblib' or
    'easy_install -U joblib'.
    """

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'))
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

        # download archive
        url = self.URL
        sha1 = self.SHA1
        basename = path.basename(url)
        archive_filename = path.join(home, basename)
        if not path.exists(archive_filename):
            if not download_if_missing:
                return
            if not path.exists(home):
                os.makedirs(home)
            download(url, archive_filename, sha1=sha1)

        # extract it
        if not path.exists(self.home(self.SUBDIR)):
            extract(archive_filename, home, sha1=sha1, verbose=True)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if hasattr(self, '_meta'):
            return self._meta
        else:
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
            return self._meta

    def _get_meta(self):

        names = sorted(os.listdir(self.home(self.SUBDIR)))

        meta = []
        ind = 0

        for name in names:

            pattern = self.home(self.SUBDIR, name, '*.jpg')

            img_filenames = sorted(glob(pattern))

            for img_filename in img_filenames:
                img_data = open(img_filename, 'rb').read()
                sha1 = hashlib.sha1(img_data).hexdigest()

                data = dict(name=name,
                            id=ind,
                            filename=img_filename,
                            sha1=sha1)

                meta += [data]
                ind += 1

        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Standard Tasks
    # ------------------------------------------------------------------------

    def raw_classification_task(self):
        """Return image_paths, labels"""
        image_paths = [m['filename'] for m in self.meta]
        names = np.asarray([m['name'] for m in self.meta])
        labels = int_labels(names)
        return image_paths, labels

    def img_classification_task(self, dtype='uint8'):
        img_paths, labels = self.raw_classification_task()
        imgs = larray.lmap(ImgLoader(ndim=3, dtype=dtype, mode='RGB'),
                           img_paths)
        return imgs, labels


class Caltech101(BaseCaltech):
    URL = ('http://www.vision.caltech.edu/Image_Datasets/'
           'Caltech101/101_ObjectCategories.tar.gz')
    SHA1 = 'b8ca4fe15bcd0921dfda882bd6052807e63b4c96'
    SUBDIR = '101_ObjectCategories'


class Caltech256(BaseCaltech):
    URL = ('http://www.vision.caltech.edu/Image_Datasets/'
           'Caltech256/256_ObjectCategories.tar')
    SHA1 = '2195e9a478cf78bd23a1fe51f4dabe1c33744a1c'
    SUBDIR = '256_ObjectCategories'
