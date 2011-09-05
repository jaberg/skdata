# -*- coding: utf-8 -*-

"""PASCAL Visual Object Classes (VOC) Datasets

http://pascallin.ecs.soton.ac.uk/challenges/VOC

If you make use of this data, please cite the following journal paper in
any publications:

The PASCAL Visual Object Classes (VOC) Challenge
Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman,
A.  International Journal of Computer Vision, 88(2), 303-338, 2010

http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.pdf
http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.html#abstract
http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.html#bibtex
"""

# Copyright (c) 2011 Nicolas Pinto <pinto@rowland.harvard.edu>
# Copyright (c) 2011 Nicolas Poilvert <poilvert@rowland.harvard.edu>
# License: Simplified BSD

import os
from os import path
import shutil
from glob import glob

import logging
logger = logging.getLogger(__name__)

import numpy as np

from data_home import get_data_home
from utils import download, extract, xml2list


class BasePASCAL(object):

    def __init__(self, meta=None):
        """PASCAL VOC Dataset

        Attributes
        ----------
        meta: list of dict
            Metadata associated with the dataset. For each image with index i,
            meta[i] is a dict with keys:

                id: str
                    Identifier of the image.

                filename: str
                    Full path to the image.

                shape: tuple
                    Shape of the image (height, width, depth).

                split: str
                    'train', 'val' or 'test'.

                objects: list of dict [optional]
                    Description of the objects present in the image. Note that
                    this key may not be available if split is 'test'. If the
                    key is present, then objects[i] is a dict with keys:

                        name: str
                            Name (label) of the object.

                        bounding_box: tuple of int
                            Bounding box coordinates (0-based index) of the
                            form (x_min, x_max, y_min, y_max):
                            +-----------------------------------------▶ x-axis
                            |
                            |   +-------+    .  .  .  y_min (top)
                            |   | bbox  |
                            |   +-------+    .  .  .  y_max (bottom)
                            |
                            |   .       .
                            |
                            |   .       .
                            |
                            |  x_min   x_max
                            |  (left)  (right)
                            |
                            ▼
                            y-axis

                        pose: str
                            'Left', 'Right', 'Frontal', 'Rear' or 'Unspecified'

                        truncated: boolean
                            True if the object is occluded / truncated.

                        difficult: boolean
                            True if the object has been tagged as difficult
                            (should be ignored during evaluation?).

                segmented: boolean
                    True if segmentation information is available.

                owner: dict
                    Owner of the image (self-explanatory).

                source: dict
                    Source of the image (self-explanatory).


        Notes
        -----
        If joblib is available, then `meta` be cached for faster processing. To
        install joblib use 'pip install -U joblib' or 'easy_install -U joblib'.
        """

        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'))
            self._build_meta = mem.cache(self._build_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), 'pascal', self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self):

        home = self.home()
        if not path.exists(home):
            os.makedirs(home)

        # download archives
        archive_filenames = []
        for url in self.ARCHIVES.itervalues():
            basename = path.basename(url)
            archive_filename = path.join(home, basename)
            if not path.exists(archive_filename):
                download(url, archive_filename)
            archive_filenames += [archive_filename]

        # extract them
        if not path.exists(path.join(home, 'VOCdevkit')):
            for archive_filename in archive_filenames:
                extract(archive_filename, home, verbose=True)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if hasattr(self, '_meta'):
            return self._meta
        else:
            self.fetch()
            self._meta = self._build_meta()
            return self._meta

    def _build_meta(self):

        base_dirname = self.home('VOCdevkit', self.name)
        dirs = dict([(basename, path.join(base_dirname, basename))
                      for basename in os.listdir(base_dirname)
                      if path.isdir(path.join(base_dirname, basename))])

        img_pattern = path.join(dirs['JPEGImages'], "*.jpg")
        img_filenames = glob(img_pattern)
        img_filenames.sort()
        n_imgs = len(img_filenames)

        # --
        print "Parsing annotations..."
        meta = []
        unique_object_names = []
        n_objects = 0
        img_ids = []
        for ii, img_filename in enumerate(img_filenames):

            data = {}

            data['filename'] = img_filename

            img_basename = path.basename(path.split(img_filename)[1])
            img_id = path.splitext(img_basename)[0]
            img_ids += [img_id]

            data['id'] = img_id

            # -- get xml filename
            xml_filename = path.join(dirs['Annotations'],
                                     "%s.xml" % img_id)
            if not path.exists(xml_filename):
                # annotation missing
                meta += [data]
                continue

            # -- parse xml
            xl = xml2list(xml_filename)

            # image basename
            assert img_basename == xl[1]

            # source
            data['source'] = xl[2]

            # owner
            data['owner'] = xl[3]

            # size / shape
            size = xl[4]
            data['shape'] = size['height'], size['width'], size['depth']

            # segmentation ?
            segmented = bool(xl[5])
            data['segmented'] = segmented
            if segmented:
                # TODO: parse segmentation data (in 'SegmentationClass') or
                # lazy-evaluate it ?
                pass

            # objects
            objs = xl[6:]
            objects = []
            for obj in objs:
                bndbox = obj.pop('bndbox')
                bounding_box = [(int(bndbox[key]) - 1)
                                for key in 'xmin', 'xmax', 'ymin', 'ymax']
                obj['bounding_box'] = tuple(bounding_box)
                n_objects += 1
                if obj['name'] not in unique_object_names:
                    unique_object_names += [obj['name']]
                objects += [obj]
            data['objects'] = objects

            # append to meta
            meta += [data]

            # -- print progress
            n_done = ii + 1
            status = ("Progress: %d/%d [%.1f%%]"
                      % (n_done, len(img_filenames), 100. * n_done / n_imgs))
            status += chr(8) * (len(status) + 1)
            print status,

            #if n_done == 100: break # DEBUG

        print

        print " Number of images: %d" % len(meta)
        print " Number of unique object names: %d" % len(unique_object_names)
        print " Unique object names: %s" % unique_object_names

        # --
        print "Parsing splits..."
        main_dirname = path.join(dirs['ImageSets'], 'Main')

        # We use 'aeroplane_{train,trainval}.txt' to get the list of 'train'
        # and 'val' ids
        train_filename = path.join(main_dirname, 'aeroplane_train.txt')
        assert path.exists(train_filename)
        train_ids = np.loadtxt(train_filename, dtype=str)[:, 0]

        trainval_filename = path.join(main_dirname, 'aeroplane_trainval.txt')
        assert path.exists(trainval_filename)
        trainval_ids = np.loadtxt(trainval_filename, dtype=str)[:, 0]

        splits = 'train', 'val', 'test'
        split_counts = dict([(split, 0) for split in splits])
        for data in meta:
            img_id = data['id']
            #if img_id >= 100: break #@ DEBUG
            if img_id in trainval_ids:
                if img_id in train_ids:
                    data['split'] = 'train'
                else:
                    data['split'] = 'val'
            else:
                data['split'] = 'test'
            split_counts[data['split']] += 1

        for split in splits:
            count = split_counts[split]
            assert count > 0
            print(" Number of images in '%s': %d"
                  % (split, count))

        meta = np.array(meta)
        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home):
            shutil.rmtree(self.home)

    # ------------------------------------------------------------------------
    # -- Driver routines to be called by datasets.main
    # ------------------------------------------------------------------------

    @classmethod
    def main_fetch(cls):
        cls.fetch(download_if_missing=True)

    @classmethod
    def main_show(cls):
        raise NotImplementedError


class VOC2007(BasePASCAL):
    ARCHIVES = {
        'trainval': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/'
            'VOCtrainval_06-Nov-2007.tar'
        ),
        'test': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/'
            'VOCtest_06-Nov-2007.tar'
        ),
    }


class VOC2008(BasePASCAL):
    ARCHIVES = {
        'trainval': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2008/'
            'VOCtrainval_14-Jul-2008.tar'
        ),
        'test': (
            'https://s3.amazonaws.com/scikits.data/pascal/'
            'VOC2008test.tar'
        ),
    }


class VOC2009(BasePASCAL):
    ARCHIVES = {
        'trainval': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2009/'
            'VOCtrainval_11-May-2009.tar'
        ),
        'test': (
            'https://s3.amazonaws.com/scikits.data/pascal/'
            'VOC2009test.tar'
        ),
    }


class VOC2010(BasePASCAL):
    ARCHIVES = {
        'trainval': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/'
            'VOCtrainval_03-May-2010.tar'
        ),
        'test': (
            'https://s3.amazonaws.com/scikits.data/pascal/'
            'VOC2010test.tar'
        ),
    }


class VOC2011(BasePASCAL):
    ARCHIVES = {
        'trainval': (
            'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2011/'
            'VOCtrainval_25-May-2011.tar'
        ),
        'test': (
            'https://s3.amazonaws.com/scikits.data/pascal/'
            'VOC2010test.tar.gz'
        ),
    }


def main_fetch():
    raise NotImplementedError


def main_show():
    raise NotImplementedError
