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
# License: Simplified BSD


# ISSUES (XXX)
# - Extra pairs.txt files in the funneled dataset.  The lfw-funneled.tgz dataset
#   has, in the same dir as the names, a bunch of pairs_0N.txt files and a
#   pairs.txt file. Why are they there?  Should we be using them?
# - self.descr should be used
# - self.meta_const should be used to store the image shapes

import logging
import os
from os import listdir, makedirs, remove
from os.path import exists, isdir
import shutil
import sys
import tarfile
import urllib

import lockfile  # available from pypi
import numpy as np

from data_home import get_data_home
import larray
import utils
import utils.image

logger = logging.getLogger(__name__)


class BaseLFW(object):
    """This base class handles both the original and funneled datasets.

    The lfw subdirectory in the datasets cache has the following structure, when
    it has been populated by calling `funneled.fetch()` and `original.fetch()`.

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

    Fetching the data is easily done using the datasets-fetch utility. In bash
    type

    .. code-block:: bash

        datasets-fetch lfw.original  # downloads and untars original dataset
        datasets-fetch lfw.funneled  # downloads and untars funneled dataset

    The meta attribute of this class is a list of dictionaries.
    Each dictionary describes one image in the dataset, with the following keys:
        name: string (e.g. Adam_Sandler)
        number: int (4 means filename is Adam_Sandler_0004.jpg)
        pairs: dict
            split_name: list of ints

    The list of ints associated with a set_name is the pair_id numbers in which
    this image is an element.

    """

    PAIRS_BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
    PAIRS_FILENAMES = [
        'pairsDevTrain.txt',
        'pairsDevTest.txt',
        'pairs.txt',
    ]

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta

    #
    # Standard dataset object interface
    #

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=True)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    #
    # Helper routines
    #

    def build_meta(self):
        pairs = {}
        pairs['DevTrain'] = self.parse_pairs('pairsDevTrain.txt')[0]
        pairs['DevTest'] = self.parse_pairs('pairsDevTest.txt')[0]
        for i, fold_i in enumerate(self.parse_pairs('pairs.txt')):
            pairs['fold_%i' % i] = fold_i
        assert i == getattr(self, 'N_PAIRS_SPLITS', i)
        meta = []
        for name in sorted(listdir(self.home(self.IMAGEDIR))):
            if isdir(self.home(self.IMAGEDIR, name)):
                for filename in sorted(listdir(self.home(self.IMAGEDIR, name))):
                    number = int(filename[-8:-4])
                    assert filename == '%s_%04i.jpg' % (name, number)
                    dct = dict(name=name, number=number, pairs={})
                    for set_name, set_dct in pairs.items():
                        dct['pairs'][set_name] = set_dct.get((name, number), [])
                    meta.append(dct)
        return meta

    def parse_pairs(self, txt_relpath):
        """
        index_file is a text file whose rows are one of
            name1 I name2 J   # non-matching pair    (name1, I), (name2, J)
            name I J          # matching image pair  (name, I), (name, J)
            N                 # line will be followed by N matching pairs and
                              # then followed by N non-matching pairs
            N M               # line will be followed by N matching pairs and
                              # then N non-matching pairs... M times

        This function returns a list of dictionaries mapping (name, I) pairs to
        the ids of image pairs in which (name, I) appears.
        There is one dictionary for each fold in the file.
        """
        txtfile = open(self.home(txt_relpath), 'rb')
        header = txtfile.readline().strip().split('\t')
        lines = [l.strip().split('\t') for l in txtfile.readlines()]
        lines_iter = iter(lines)
        try:
            n_splits, n_match_per_split = [int(h) for h in header]
        except ValueError:
            n_splits, n_match_per_split = [1] + [int(h) for h in header]
        rval = []
        for split_idx in range(n_splits):
            dct = {}
            try:
                for i in xrange(n_match_per_split):
                    tmp = lines_iter.next()
                    name, I, J = tmp
                    dct.setdefault((name, int(I)), []).append(i)
                    dct.setdefault((name, int(J)), []).append(i)
                for i in xrange(n_match_per_split, 2 * n_match_per_split):
                    tmp = lines_iter.next()
                    name1, I, name2, J = tmp
                    dct.setdefault((name1, int(I)), []).append(i)
                    dct.setdefault((name2, int(J)), []).append(i)
            except ValueError:
                print "Parse error on line:", tmp
                raise
            rval.append(dct)
        return rval

    def image_path(self, dct):
        return self.home(
                self.IMAGEDIR,
                dct['name'],
                '%s_%04i.jpg' % (dct['name'], dct['number']))

    #
    # Fetch interface (XXX is this a general interface?)
    #

    def home(self, *names):
        return os.path.join(get_data_home(), 'lfw', self.NAME, *names)

    def clean_up(self):
        if isdir(self.home()):
            shutil.rmtree(self.home())

    def fetch(self, download_if_missing=True):
        """Download the dataset if necessary."""

        if not exists(self.home()):
            makedirs(self.home())

        lock = lockfile.FileLock(self.home())
        if lock.is_locked():
            logger.warn('%s is locked, waiting for release' %
                    self.home())
        with lock:

            # download the little metadata .txt files
            for target_filename in self.PAIRS_FILENAMES:
                target_filepath = self.home(target_filename)
                if not exists(target_filepath):
                    if download_if_missing:
                        url = self.PAIRS_BASE_URL + target_filename
                        logger.warn("Downloading LFW metadata: %s => %s" % (
                            url, target_filepath))
                        downloader = urllib.urlopen(url)
                        data = downloader.read()
                        open(target_filepath, 'wb').write(data)
                    else:
                        raise IOError("%s is missing" % target_filepath)

            if not exists(self.home(self.IMAGEDIR)):
                archive_path = self.home('images.tgz')
                # download the tgz
                if not exists(archive_path):
                    if download_if_missing:
                        logger.warn(
                                "Downloading LFW data (~200MB): %s => %s" % (
                                self.URL, archive_path))
                        downloader = urllib.urlopen(self.URL)
                        data = downloader.read()
                        # don't open file until download is complete
                        open(archive_path, 'wb').write(data)
                    else:
                        raise IOError("%s is missing" % target_filepath)

                logger.info("Decompressing the data archive to %s",
                        self.home())
                tarfile.open(archive_path, "r:gz").extractall(
                        path=self.home())
                remove(archive_path)

    #
    # Driver routines to be called by datasets.main
    #
    @classmethod
    def main_fetch(cls):
        """compatibility with bin/datasets_fetch"""
        cls.fetch(download_if_missing=True)

    @classmethod
    def main_show(cls):
        # Usage one of:
        # <driver> people
        # <driver> pairs
        from utils.glviewer import glumpy_viewer
        try:
            task = sys.argv[2]
        except IndexError:
            print >> sys.stderr, "Usage one of"
            print >> sys.stderr, "    <driver> lfw.<imgset> people"
            print >> sys.stderr, "    <driver> lfw.<imgset> pairs"
            print >> sys.stderr, "    <driver> lfw.<imgset> pairs_train"
            print >> sys.stderr, "    <driver> lfw.<imgset> pairs_test"
            print >> sys.stderr, "    <driver> lfw.<imgset> pairs_10folds"
            return 1

        if task == 'people':
            self = cls()
            image_paths = [self.image_path(m) for m in self.meta]
            names = np.asarray([m['name'] for m in self.meta])
            glumpy_viewer(
                    img_array=larray.lmap(
                        utils.image.load_rgb_f32,
                        image_paths),
                    arrays_to_print=[names])
        elif task == 'pairs' or sys.argv[2] == 'pairs_train':
            raise NotImplementedError()
        elif task == 'pairs_test':
            raise NotImplementedError()
        elif task == 'pairs_10folds':
            raise NotImplementedError()

    #
    # Standard tasks built from self.meta
    # -----------------------------------
    #
    # raw_... methods return filename lists
    # img_... methods return lazily-loaded image lists
    #

    def raw_classification_task(self):
        """Return image_paths, labels"""
        image_paths = [self.image_path(m) for m in self.meta]
        names = np.asarray([m['name'] for m in self.meta])
        labels = utils.int_labels(names)
        return image_paths, labels

    def raw_verification_task_view2(self, split_role, split_k):
        """Return a train or test split from View 2

        :param split_role: either 'train' or 'test'
        :param split_k: an integer from 0 to 9 inclusive.

        :param rtype: [optional] a callable that casts the return value (for
                      internal use)

        :returns: Return left_image_paths, right_image_paths, labels of LFW
        "View 2" that is to be used for testing.

        """
        split_k = int(split_k)
        if split_k not in range(10):
            raise ValueError(split_k)
        if split_role not in ('train', 'test'):
            raise ValueError(split_role)
        if split_role == 'test':
            return self.raw_verification_task(split='fold_%i' % split_k)
        else:
            # -- split_role is 'train'
            L, R, Y = [], [], []
            for k in range(10):
                if k == split_k:
                    continue
                Lk, Rk, Yk = self.raw_verification_task('fold_%i' % k)
                L += list(Lk)
                R += list(Rk)
                Y += list(Yk)
            return (np.asarray(L),
                    np.asarray(R),
                    np.asarray(Y, dtype='int'))

    def raw_verification_task(self, split='DevTrain'):
        """Return left_image_paths, right_image_paths, labels

        :param split: one of 'DevTrain', 'DevTest', 'fold_0', 'fold_1', ...
                    'fold_9'.

        DevTrain returns the "View 1" training data. DevTest returns the
        "View 1" testing data.  If split is `fold_k`, then this function
        returns the test data of the k'th split from the "View 2" set.
        """
        paths = {}
        if split not in ('DevTrain', 'DevTest',
                'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5',
                'fold_6', 'fold_7', 'fold_8', 'fold_9'):
            raise KeyError('invalid split', split)
        for m in self.meta:
            for pid in m['pairs'][split]:
                paths.setdefault(pid, []).append(m)
        ids = range(max(paths.keys()) + 1)
        # begin sanity checking
        for mlist in paths.values():
            assert len(mlist) == 2
        assert len(ids) == len(paths)
        # end sanity checking
        left_image_paths = [self.image_path(paths[i][0]) for i in ids]
        right_image_paths = [self.image_path(paths[i][1]) for i in ids]
        labels = [paths[i][0]['name'] == paths[i][1]['name'] for i in ids]
        return (np.asarray(left_image_paths),
                np.asarray(right_image_paths),
                np.asarray(labels, dtype='int'))

    def img_classification_task(self, dtype='uint8'):
        img_paths, labels = self.raw_classification_task()
        imgs = larray.lmap(
                utils.image.ImgLoader(shape=self.img_shape, dtype=dtype),
                img_paths)
        return imgs, labels

    def img_verification_task_from_raw(self, lpaths, rpaths, labels,
            dtype='uint8'):
        limgs = larray.lmap(
                utils.image.ImgLoader(shape=self.img_shape, dtype=dtype),
                lpaths)
        rimgs = larray.lmap(
                utils.image.ImgLoader(shape=self.img_shape, dtype=dtype),
                rpaths)
        return limgs, rimgs, labels

    def img_verification_task(self, split='DevTrain', dtype='uint8'):
        """
        Return (left images, right images, labels) for the given split.
        """
        lpaths, rpaths, labels = self.raw_verification_task(split)
        return self.img_verification_task_from_raw(lpaths, rpaths, labels,
                dtype=dtype)


class Original(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    NAME = 'original'  # self.home() is <CACHE>/lfw/<NAME>
    IMAGEDIR = 'lfw'   # this matches what comes out of the tgz
    img_shape = (250, 250, 3)


class Funneled(BaseLFW):
    URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    NAME = 'funneled'          # self.home() is <CACHE>/lfw/<NAME>
    IMAGEDIR = 'lfw_funneled'  # this matches what comes out of the tgz
    img_shape = (250, 250, 3)


class Aligned(BaseLFW):
    URL = "http://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz"
    NAME = 'aligned'          # self.home() is <CACHE>/lfw/<NAME>
    IMAGEDIR = 'lfw2'         # this matches what comes out of the tgz
    img_shape = (250, 250)


def main_fetch():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")


def main_show():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")
