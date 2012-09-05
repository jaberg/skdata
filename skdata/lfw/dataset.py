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
most popular model for Face Detection is called Viola-Jones and is
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
from glob import glob
import shutil

import lockfile
import numpy as np

from skdata.data_home import get_data_home
from skdata.utils import download, download_and_extract

import logging
log = logging.getLogger(__name__)
# XXX: logging config (e.g. formatting, etc.) should be factored out in
# skdata and not be imposed on the caller (like in http://goo.gl/7xEeB)


NAMELEN = 48

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

    The meta-data is dictionaries (one-per-image) with keys:
    * filename
    * name
    * image_number

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
                os.makedirs(output_dirname)

            # -- various disruptions might cause this to fail
            #    but if any process gets as far as writing the completion
            #    marker, then it should be all good.
            done_marker = os.path.join(output_dirname, 'completion_marker')
            if not path.exists(done_marker):
                download_and_extract(url, output_dirname, sha1=sha1)
                open(done_marker, 'w').close()

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

    def parse_pairs_file(self, filename):
        """
        Return recarray of n_folds x n_labels x n_pairs x 2

        There are 2 labels: label 0 means same, label 1 means different

        Each element of the recarray has two fields: 'name' and 'inum'.
        - The name is the name of the person in the LFW picture.
        - The inum is the number indicating which LFW picture of the person
          should be used.
        """
        self.fetch()

        # -- load the pairs/labels txt file into one string per line
        lines = np.loadtxt(filename, dtype=str, delimiter='\n')
        header = lines[0].split()
        header_tokens = map(int, header)
        if len(header_tokens) == 2:
            n_folds, n_pairs = header_tokens
        elif len(header_tokens) == 1:
            n_folds, n_pairs = [1] + header_tokens
        else:
            raise ValueError('Failed to parse header', header_tokens)

        # -- checks number of lines by side-effect
        elems = lines[1:].reshape(n_folds, 2, n_pairs)
        rval = np.recarray((n_folds, 2, n_pairs, 2),
                dtype=np.dtype([('name', 'S%i' % NAMELEN),
                    ('inum', np.int32)]))

        for fold_i in xrange(n_folds):
            # parse the same-name lines
            for pair_i in xrange(n_pairs):
                name, inum0, inum1 = elems[fold_i, 0, pair_i].split()
                assert len(name) < NAMELEN
                rval[fold_i, 0, pair_i, 0] = name, int(inum0)
                rval[fold_i, 0, pair_i, 1] = name, int(inum1)

                assert rval[fold_i, 0, pair_i, 0]['name'] == name

            # parse the different-name lines
            for pair_i in xrange(n_pairs):
                name0, inum0, name1, inum1 = elems[fold_i, 1, pair_i].split()
                assert len(name0) < NAMELEN
                assert len(name1) < NAMELEN
                rval[fold_i, 1, pair_i, 0] = name0, int(inum0)
                rval[fold_i, 1, pair_i, 1] = name1, int(inum1)

        return rval

    @property
    def pairsDevTrain(self):
        return self.parse_pairs_file(self.home('pairsDevTrain.txt'))

    @property
    def pairsDevTest(self):
        return self.parse_pairs_file(self.home('pairsDevTest.txt'))

    @property
    def pairsView2(self):
        return self.parse_pairs_file(self.home('pairs.txt'))


class Original(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    SHA1 = '1aeea1f6b1cfabc8a0e103d974b590fda315e147'
    IMAGE_SUBDIR = 'lfw'
    COLOR = True


class Funneled(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    SHA1 = '7f5c008acbd96597ee338fbb2d6c0045979783f7'
    IMAGE_SUBDIR = 'lfw_funneled'
    COLOR = True


class Aligned(BaseLFW):
    URL = "http://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz"
    SHA1 = '38ecda590870e7dc91fb1040759caccddbe25375'
    IMAGE_SUBDIR = 'lfw2'
    COLOR = False
