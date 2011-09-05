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
        """PASCAL VOC Dataset Object

        Use '.meta' to access the metadata. Note that if joblib is available,
        the metadata will be cached for faster processing. To install joblib
        use 'pip install -vU joblib'.
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

            # size (width, height, depth)
            data.update(xl[4])

            # segmentation ?
            segmented = bool(xl[5])
            data['segmented'] = segmented
            if segmented:
                # TODO: parse segmentation data (in 'SegmentationClass')
                pass

            # objects
            # XXX: change the names of the keys, eg 'bndbox' -> 'bounding_box'
            objs = xl[6:]
            data['objects'] = objs
            for obj in objs:
                n_objects += 1
                if obj['name'] not in unique_object_names:
                    unique_object_names += [obj['name']]

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
            print(" Number of images in '%s': %d"
                  % (split, split_counts[split]))

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
