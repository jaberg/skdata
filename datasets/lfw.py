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
import img_utils

logger = logging.getLogger(__name__)

# Create two image loader functions for lmap.
# This way the dataset can promise what size the images will be
# while still allowing them to be loaded lazily.
img_loader_u8 = img_utils.ImgLoader(shape=(250, 250, 3), dtype='uint8')
img_loader_f32 = img_utils.ImgLoader(shape=(250, 250, 3), dtype='float32')


class BaseLFW(object):
    """
    This base class handles both the original and funneled datasets.

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

    """
    BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
    TARGET_FILENAMES = [
        'pairsDevTrain.txt',
        'pairsDevTest.txt',
        'pairs.txt',
    ]

    def home(cls, *names):
        return join(get_data_home(), 'lfw', cls.NAME, *names)

    def erase(cls):
        if isdir(cls.home()):
            shutil.rmtree(cls.home())

    def disksize(cls):
        raise NotImplementedError()

    def fetch(cls, download_if_missing=True):
        """Download the funneled or non-funneled dataset, if necessary.

        Call this function with no arguments to download the funneled LFW dataset to the standard
        location. This downloads about 200MB.

        If the dataset has already been downloaded, this function returns
        immediately.

        """

        archive_path = cls.home(cls.ARCHIVE_NAME)
        images_root = cls.home('images')
        archive_url = cls.BASE_URL + cls.ARCHIVE_NAME

        if not exists(cls.home()):
            makedirs(cls.home())

        # download the little metadata .txt files
        for target_filename in cls.TARGET_FILENAMES:
            target_filepath = join(cls.home(), target_filename)
            if not exists(target_filepath):
                if download_if_missing:
                    url = cls.BASE_URL + target_filename
                    logger.warn("Downloading LFW metadata: %s => %s" % (
                        url, target_filepath))
                    downloader = urllib.urlopen(cls.BASE_URL + target_filename)
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

    def image_lists(cls):
        """Return a dictionary mapping names (w underscores) to a list of
        filenames of images of that person.

        This routine walks the data_folder_path to build these lists.
        """
        rval = {}
        for person_name in listdir(cls.home(cls.IMAGEDIR)):
            rval[person_name] = [cls.home(cls.IMAGEDIR, person_name, filename)
                    for filename in sorted(listdir(
                        cls.home(cls.IMAGEDIR, person_name)))]
        return rval

    def image_pairs(cls, txt_relpath):
        """
        index_file is a text file whose rows are one of
            name1 I name2 J   # non-matching pair    (name1, I), (name2, J)
            name I J          # matching image pair  (name, I), (name, J)
            N                 # line will be followed by N matching pairs and
                              # then followed by N non-matching pairs
            N M               # line will be followed by N matching pairs and
                              # then N non-matching pairs... M times

        This function returns three lists:
        - left image paths
        - right image paths
        - binary ndarray: targets match?
        """
        txtfile = open(cls.home(txt_relpath), 'rb')
        splitted_lines = [l.strip().split('\t') for l in txtfile.readlines()]
        pair_specs = [sl for sl in splitted_lines if len(sl) > 2]
        n_pairs = len(pair_specs)

        # interating over the metadata lines for each pair to find the filename to
        # decode and load in memory
        target = np.zeros(n_pairs, dtype=np.int)
        left_paths = []
        right_paths = []
        person_names, file_paths = names_paths(data_folder_path, 0)
        img_lists = cls.image_lists()

        for i, components in enumerate(pair_specs):
            if len(components) == 3:
                target[i] = 1
                left = (components[0], int(components[1]) - 1)
                right = (components[0], int(components[2]) - 1)
            elif len(components) == 4:
                target[i] = 0
                left = (components[0], int(components[1]) - 1)
                right = (components[2], int(components[3]) - 1)
            else:
                raise ValueError("invalid line %d: %r" % (i + 1, components))

            # a dictionary would make this more readable.
            left_paths.append(img_lists[left[0]][left[1]])
            right_paths.append(img_lists[right[0]][right[1]])

        return left_paths, right_paths, target

    def load_people(cls, download_if_missing=True):
        """
        Return Bunch with keys:
            names: 1-d ndarray of names of people
            img_fullpath: 1-d ndarray of paths to pictures of them
            labels: like names, but integers instead of strings
            DESCR: "LFW faces dataset"
        """
        cls.fetch(download_if_missing)
        img_lists = cls.image_lists()
        names, filenames = [], []
        for name in sorted(img_lists):
            for fname in img_lists[name]:
                names.append(name)
                filenames.append(fname)
        names = np.asarray(names)
        filenames = np.asarray(filenames)
        labels = np.searchsorted(sorted(img_lists), names)
        assert len(names) == len(labels) == len(filenames)
        return Bunch(
                names=names,
                images=larray.lmap(img_loader_u8, filenames),
                images_f32=larray.lmap(img_loader_f32, filenames),
                img_fullpath=filenames,
                labels=labels,
                DESCR="LFW faces dataset")

    def load_pairs_train(cls, download_if_missing=True):
        return cls._load_pairs('pairsDevTrain.txt', 'train',
                download_if_missing)

    def load_pairs_test(cls, download_if_missing=True):
        return cls._load_pairs('pairsDevTest.txt', 'test',
                download_if_missing)

    def load_pairs_10folds(cls, download_if_missing=True):
        return cls._load_pairs('pairs.txt', '10folds',
                download_if_missing)

    def _load_pairs(cls, txt_relpath, subset_name, download_if_missing):
        cls.fetch(download_if_missing)
        lnames, rnames, lpaths, rpaths, labels = image_pairs(txt_relpath)
        return Bunch(
                lnames=lnames,
                rnames=rnames,
                lpaths=lpaths,
                rpaths=rpaths,
                labels=labels,
                DESCR="'%s' segment of the LFW pairs dataset" % subset_name)

    def main_fetch(cls):
        """compatibility with bin/datasets_fetch"""
        cls.fetch(download_if_missing=True)

    def main_show(cls):
        # Usage one of:
        # <driver> people
        # <driver> pairs
        from glviewer import glumpy_viewer, command, glumpy
        import larray
        import img_utils
        print 'ARGV', sys.argv
        if sys.argv[2] == 'people':
            bunch = cls.load_people()
            glumpy_viewer(
                    img_array=larray.lmap(
                        img_utils.read_rgb_float32,
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


class Original(BaseLFW):
    NAME = 'original'
    ARCHIVE_NAME = "lfw.tgz"


class Funneled(BaseLFW):
    NAME = 'funneled'
    ARCHIVE_NAME = "lfw-funneled.tgz"


def main_fetch():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")


def main_show():
    raise NotImplementedError(
            "Please specify either lfw.Funneled or lfw.Original")

