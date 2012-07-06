"""Loader for the Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/

Each picture is centered on a single face. The typical task is called
Face Verification: given a pair of two pictures, a binary classifier
must predict whether the two images are from the same person.

An alternative task, Face Recognition or Face Identification is:
given the picture of the face of an unknown person, identify the name
of the person by refering to a gallery of previously seen pictures of
identified persons.

Both Face Verification and Face Recognition are tasks that are typically
performed on the output of a model trained to perform Face Detection. The
most popular model for Face Detection is called Viola-Johns and is
implemented in the OpenCV library. The LFW faces were extracted by this face
detector from various online websites.
"""

# Copyright (c) 2011 James Bergstra <bergstra@rowland.harvard.edu>
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# Copyright (c) 2012 Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD


# ISSUES (XXX)
# - Extra pairs.txt files in the funneled dataset.  The lfw-funneled.tgz dataset
#   has, in the same dir as the names, a bunch of pairs_0N.txt files and a
#   pairs.txt file. Why are they there?  Should we be using them?

import os
from os import path
from os import listdir, makedirs, remove
from os.path import exists, isdir
from glob import glob
import shutil
#import sys
import tarfile
import urllib

import lockfile
import numpy as np

from skdata.data_home import get_data_home
from skdata.utils import download, download_and_extract
#import larray
#import utils
#import utils.image

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# XXX: logging config (e.g. formatting, etc.) should be factored out in
# skdata and not be imposed on the caller (like in http://goo.gl/7xEeB)


PAIRS_BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
PAIRS_FILENAMES = [
    ('pairsDevTrain.txt', '082b7adb005fd30ad35476c18943ce66ab8ff9ff'),
    ('pairsDevTest.txt', 'f33ea17f58dac4401801c5c306f81d9ff56e30e9'),
    ('pairs.txt', '020efa51256818a30d3033a98fc98b97a8273df2'),
]


class BaseLFW(object):
    """XXX

    The lfw subdirectory in the datasets cache has the following structure, when
    it has been populated by calling `fetch()`.

    .. code-block::

        lfw/
            funneled/
                pairs.txt
                pairsDevTrain.txt
                pairsDevTrain.txt
                images/
                    lfw_funneled/
                        <names>/
                            <jpgs>
            original/
                pairs.txt
                pairsDevTrain.txt
                pairsDevTrain.txt
                images/
                    lfw/
                        <names>/
                            <jpgs>

            ...

    """

    def __init__(self):

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'), verbose=False)
            self._get_meta = mem.cache(self._get_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), 'lfw', self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self):
        """Download and extract the dataset."""

        home = self.home()

        lock = lockfile.FileLock(home)
        if lock.is_locked():
            log.warn('%s is locked, waiting for release' % home)

        with lock:
            # -- download pair labels
            for fname, sha1 in PAIRS_FILENAMES:
                url = path.join(PAIRS_BASE_URL, fname)
                basename = path.basename(url)
                filename = path.join(home, basename)
                if not path.exists(filename):
                    if not path.exists(home):
                        os.makedirs(home)
                    download(url, filename, sha1=sha1)

            # -- download and extract images
            url = self.URL
            sha1 = self.SHA1
            output_dirname = self.home('images')
            if not path.exists(output_dirname):
                if not path.exists(output_dirname):
                    os.makedirs(output_dirname)
                download_and_extract(url, output_dirname, sha1=sha1)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta

    def _get_meta(self):

        log.info('Building metadata...')

        # -- Filenames
        pattern = self.home('images', self.IMAGE_SUBDIR, '*', '*.jpg')
        fnames = sorted(glob(pattern))
        n_images = len(fnames)
        log.info('# images = %d' % n_images)

        meta = []
        for fname in fnames:
            name = path.split(path.split(fname)[0])[-1]
            image_number = int(path.splitext(path.split(fname)[-1])[0][-4:])
            data = dict(filename=fname, name=name, image_number=image_number)
            meta += [data]

        return np.array(meta)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- LFW Specific
    # ------------------------------------------------------------------------

    @property
    def view2_folds(self):

        self.fetch()

        # -- View2: load the pairs/labels txt file
        fname = self.home('pairs.txt')
        pairs = np.loadtxt(fname, dtype=str, delimiter='\n')
        header = pairs[0].split()
        n_folds, n_pairs = map(int, header)
        n_pairs *= 2  # n_pairs 'same' + n_pairs 'different'

        # parse the folds
        view2_folds = [[] for _ in xrange(n_folds)]
        i = 1
        for fold_i in xrange(n_folds):

            for _ in xrange(n_pairs):

                txt = pairs[i].split()
                # same
                if len(txt) == 3:
                    left = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[1]))
                    right = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[2]))
                    label = +1
                # different
                elif len(txt) == 4:
                    left = '%s/%s_%04d.jpg' % (txt[0], txt[0], int(txt[1]))
                    right = '%s/%s_%04d.jpg' % (txt[2], txt[2], int(txt[3]))
                    label = -1
                #
                else:
                    raise RuntimeError("line not understood")
                view2_folds[fold_i] += [(left, right, label)]

                i += 1

            assert len(view2_folds[fold_i]) == n_pairs

        return view2_folds


class Original(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    SHA1 = '1aeea1f6b1cfabc8a0e103d974b590fda315e147'
    NAME = 'original'
    IMAGE_SUBDIR = 'lfw'
    COLOR = True


class Funneled(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    SHA1 = '7f5c008acbd96597ee338fbb2d6c0045979783f7'
    NAME = 'funneled'
    IMAGE_SUBDIR = 'lfw_funneled'
    COLOR = True


class Aligned(BaseLFW):
    URL = "http://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz"
    SHA1 = '38ecda590870e7dc91fb1040759caccddbe25375'
    NAME = 'aligned'
    IMAGE_SUBDIR = 'lfw2'
    COLOR = False
