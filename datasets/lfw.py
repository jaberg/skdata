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

from os import listdir, makedirs, remove
from os.path import join, exists, isdir
import shutil
import sys

import logging
import numpy as np
import urllib
import tarfile

from .base import get_data_home, Bunch
import larray
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
            pairs['fold_%i'%i] = fold_i
        assert i == getattr(self, 'N_PAIRS_SPLITS', i)
        meta = []
        for name in sorted(listdir(self.home(self.IMAGEDIR))):
            for filename in sorted(listdir(self.home(self.IMAGEDIR, name))):
                number = int(filename[-8:-4])
                assert filename == '%s_%04i.jpg'%(name, number)
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
        n_match_per_split = int(header[0])
        if len(header) == 1:
            n_splits = 1
        else:
            n_splits = int(header[1])
        rval = []
        for split_idx in range(n_splits):
            dct = {}
            for i in xrange(n_match_per_split):
                name, I, J = lines_iter.next()
                dct.setdefault((name, int(I)), []).append(i)
                dct.setdefault((name, int(J)), []).append(i)
            for i in xrange(n_match_per_split, 2*n_match_per_split):
                name1, I, name2, J = lines_iter.next()
                dct.setdefault((name1, int(I)), []).append(i)
                dct.setdefault((name2, int(J)), []).append(i)
            rval.append(dct)
        return rval

    def image_path(self, dct):
        return self.home(
                self.IMAGEDIR,
                dct['name'],
                '%s_%04i.jpg'%(dct['name'], dct['number']))


    #
    # Fetch interface (XXX is this a general interface?)
    #

    def home(self, *names):
        return join(get_data_home(), 'lfw', self.NAME, *names)

    def erase(self):
        if isdir(self.home()):
            shutil.rmtree(self.home())

    def fetch(self, download_if_missing):
        """Download the funneled or non-funneled dataset, if necessary.

        Call this function with no arguments to download the funneled LFW dataset to the standard
        location. This downloads about 200MB.

        If the dataset has already been downloaded, this function returns
        immediately.

        """

        archive_path = self.home(self.ARCHIVE_NAME)
        images_root = self.home('images')
        archive_url = self.BASE_URL + self.ARCHIVE_NAME

        if not exists(self.home()):
            makedirs(self.home())

        # download the little metadata .txt files
        for target_filename in self.TARGET_FILENAMES:
            target_filepath = join(self.home(), target_filename)
            if not exists(target_filepath):
                if download_if_missing:
                    url = self.BASE_URL + target_filename
                    logger.warn("Downloading LFW metadata: %s => %s" % (
                        url, target_filepath))
                    downloader = urllib.urlopen(self.BASE_URL + target_filename)
                    data = downloader.read()
                    open(target_filepath, 'wb').write(data)
                else:
                    raise IOError("%s is missing" % target_filepath)

        if not exists(images_root):
            # download the tgz
            if not exists(archive_path):
                if download_if_missing:
                    logger.warn("Downloading LFW data (~200MB): %s => %s" %(
                            archive_url, archive_path))
                    downloader = urllib.urlopen(archive_url)
                    data = downloader.read()
                    # don't open file until download is complete
                    open(archive_path, 'wb').write(data)
                else:
                    raise IOError("%s is missing" % target_filepath)

            logger.info("Decompressing the data archive to %s", images_root)
            tarfile.open(archive_path, "r:gz").extractall(path=images_root)
            remove(archive_path)


    #
    # Driver routines to be called by datasets.main
    #

    def main_fetch(self):
        """compatibility with bin/datasets_fetch"""
        self.fetch(download_if_missing=True)

    def main_show(self):
        # Usage one of:
        # <driver> people
        # <driver> pairs
        from glviewer import glumpy_viewer, command, glumpy
        import larray
        print 'ARGV', sys.argv
        if sys.argv[2] == 'people':
            bunch = self.load_people()
            glumpy_viewer(
                    img_array=larray.lmap(
                        utils.image.read_rgb_float32,
                        bunch.img_fullpath),
                    arrays_to_print=[bunch.names])
        elif sys.argv[2] == 'pairs' or sys.argv[2] == 'pairs_train':
            raise NotImplementedError()
        elif sys.argv[2] == 'pairs_test':
            raise NotImplementedError()
        elif sys.argv[2] == 'pairs_10folds':
            fold_num = int(sys.argv[3])
            raise NotImplementedError()
        if 0:
            left_imgs = img_load(lpaths, slice_, color, resize)
            right_imgs = img_load(rpaths, slice_, color, resize)
            pairs = larray.lzip(left_imgs, right_imgs)

    #
    # Standard tasks built from self.meta
    # -----------------------------------
    #
    # raw_... methods return filename lists
    # img_... methods return lazily-loaded image lists
    #

    def raw_recognition_task(self):
        """Return image_paths, labels"""
        image_paths = [self.image_path(m) for m in self.meta]
        names = np.asarray([m['name'] for m in self.meta])
        unique_names = np.unique(names)
        labels = np.searchsorted(unique_names, names)
        return image_paths, labels

    def raw_verification_task(self, split='DevTrain'):
        """Return left_image_paths, right_image_paths, labels"""
        paths = {}
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

    def img_recognition_task(self, dtype='uint8'):
        img_paths, labels = self.raw_recognition_task()
        imgs = larray.lmap(
                utils.image.ImgLoader(shape=(250, 250, 3), dtype=dtype),
                img_paths)
        return imgs, labels

    def img_verification_task(self, split='DevTrain', dtype='uint8'):
        lpaths, rpaths, labels = self.raw_verification_task(split)
        limgs = larray.lmap(
                utils.image.ImgLoader(shape=(250, 250, 3), dtype=dtype),
                lpaths)
        rimgs = larray.lmap(
                utils.image.ImgLoader(shape=(250, 250, 3), dtype=dtype),
                rpaths)
        return limgs, rimgs, labels


class Original(BaseLFW):
    BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
    TARGET_FILENAMES = [
        'pairsDevTrain.txt',
        'pairsDevTest.txt',
        'pairs.txt',
    ]

    NAME = 'original'
    ARCHIVE_NAME = "lfw.tgz"


class Funneled(BaseLFW):
    BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
    TARGET_FILENAMES = [
        'pairsDevTrain.txt',
        'pairsDevTest.txt',
        'pairs.txt',
    ]

    NAME = 'funneled'
    ARCHIVE_NAME = "lfw-funneled.tgz"


def main_fetch():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")


def main_show():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")

