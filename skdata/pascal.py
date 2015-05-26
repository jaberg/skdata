# -*- coding: utf-8 -*-
"""PASCAL Visual Object Classes (VOC) Datasets

http://pascallin.ecs.soton.ac.uk/challenges/VOC

If you make use of this data, please cite the following journal paper in
any publication:

The PASCAL Visual Object Classes (VOC) Challenge
Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman,
A.  International Journal of Computer Vision, 88(2), 303-338, 2010

http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.pdf
http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.html#abstract
http://pascallin.ecs.soton.ac.uk/challenges/VOC/pubs/everingham10.html#bibtex
"""

# Copyright (C) 2011
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          Nicolas Poilvert <poilvert@rowland.harvard.edu>
# License: Simplified BSD

import os
from os import path
import shutil
from distutils import dir_util
from glob import glob
import hashlib

import numpy as np

from data_home import get_data_home
from utils import download, extract, xml2dict


class BasePASCAL(object):
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

            sha1: str
                SHA-1 hash of the image.

            shape: dict with int values
                Shape of the image. dict with keys 'height', 'width', 'depth'
                and int values.

            split: str
                'train', 'val' or 'test'.

            objects: list of dict [optional]
                Description of the objects present in the image. Note that this
                key may not be available if split is 'test'. If the key is
                present, then objects[i] is a dict with keys:

                    name: str
                        Name (label) of the object.

                    bounding_box: dict with int values
                        Bounding box coordinates (0-based index). dict with
                        keys 'x_min', 'x_max', 'y_min', 'y_max' and int values
                        such that:
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
                        True if the object has been tagged as difficult (should
                        be ignored during evaluation?).

            segmented: boolean
                True if segmentation information is available.

            owner: dict [optional]
                Owner of the image (self-explanatory).

            source: dict
                Source of the image (self-explanatory).


    Notes
    -----
    If joblib is available, then `meta` will be cached for faster processing.
    To install joblib use 'pip install -U joblib' or 'easy_install -U joblib'.
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
        return path.join(get_data_home(), 'pascal', self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self):
        """Download and extract the dataset."""

        home = self.home()
        if not path.exists(home):
            os.makedirs(home)

        # download archives
        archive_filenames = []
        for key, archive in self.ARCHIVES.iteritems():
            url = archive['url']
            sha1 = archive['sha1']
            basename = path.basename(url)
            archive_filename = path.join(home, basename)
            if not path.exists(archive_filename):
                download(url, archive_filename, sha1=sha1)
            archive_filenames += [(archive_filename, sha1)]
            self.ARCHIVES[key]['archive_filename'] = archive_filename

        # extract them
        if not path.exists(path.join(home, 'VOCdevkit')):
            for archive in self.ARCHIVES.itervalues():
                url = archive['url']
                sha1 = archive['sha1']
                archive_filename = archive['archive_filename']
                extract(archive_filename, home, sha1=sha1, verbose=True)
                # move around stuff if needed
                if 'moves' in archive:
                    for move in archive['moves']:
                        src = self.home(move['source'])
                        dst = self.home(move['destination'])
                        # We can't use shutil here since the destination folder
                        # may already exist. Fortunately the distutils can help
                        # us here (see standard library).
                        dir_util.copy_tree(src, dst)
                        dir_util.remove_tree(src)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if hasattr(self, '_meta'):
            return self._meta
        else:
            self.fetch()
            self._meta = self._get_meta()
            return self._meta

    def _get_meta(self):

        base_dirname = self.home('VOCdevkit', self.name)
        dirs = dict([(basename, path.join(base_dirname, basename))
                      for basename in os.listdir(base_dirname)
                      if path.isdir(path.join(base_dirname, basename))])

        img_pattern = path.join(dirs['JPEGImages'], "*.jpg")
        img_filenames = sorted(glob(img_pattern))
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

            # sha1 hash
            sha1 = hashlib.sha1(open(img_filename).read()).hexdigest()
            data['sha1'] = sha1

            # image id
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
            xd = xml2dict(xml_filename)

            # image basename
            assert img_basename == xd['filename']

            # source
            data['source'] = xd['source']

            # owner (if available)
            if 'owner' in xd:
                data['owner'] = xd['owner']

            # size / shape
            size = xd['size']
            width = int(size['width'])
            height = int(size['height'])
            depth = int(size['depth'])
            data['shape'] = dict(height=height, width=width, depth=depth)

            # segmentation ?
            segmented = bool(xd['segmented'])
            data['segmented'] = segmented
            if segmented:
                # TODO: parse segmentation data (in 'SegmentationClass') or
                # lazy-evaluate it ?
                pass

            # objects with their bounding boxes
            objs = xd['object']
            if isinstance(objs, dict):  # case where there is only one bbox
                objs = [objs]
            objects = []
            for obj in objs:
                # parse bounding box coordinates and convert them to valid
                # 0-indexed coordinates
                bndbox = obj.pop('bndbox')
                x_min = max(0,
                            (int(np.round(float(bndbox['xmin']))) - 1))
                x_max = min(width - 1,
                            (int(np.round(float(bndbox['xmax']))) - 1))
                y_min = max(0,
                            (int(np.round(float(bndbox['ymin']))) - 1))
                y_max = min(height - 1,
                            (int(np.round(float(bndbox['ymax']))) - 1))
                bounding_box = dict(x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max)
                assert (np.array(bounding_box) >= 0).all()
                obj['bounding_box'] = bounding_box
                n_objects += 1
                if obj['name'] not in unique_object_names:
                    unique_object_names += [obj['name']]

                # convert 'difficult' to boolean
                if 'difficult' in obj:
                    obj['difficult'] = bool(int(obj['difficult']))
                else:
                    # assume difficult=False if key not present
                    obj['difficult'] = False

                # convert 'truncated' to boolean
                if 'truncated' in obj:
                    obj['truncated'] = bool(int(obj['truncated']))
                else:
                    # assume truncated=False if key not present
                    obj['truncated'] = False

                objects += [obj]

            data['objects'] = objects

            # -- print progress
            n_done = ii + 1
            status = ("Progress: %d/%d [%.1f%%]"
                      % (n_done, len(img_filenames), 100. * n_done / n_imgs))
            status += chr(8) * (len(status) + 1)
            print status,

            # -- append to meta
            meta += [data]

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
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Driver routines to be called by skdata.main
    # ------------------------------------------------------------------------

    @classmethod
    def main_fetch(cls):
        cls.fetch(download_if_missing=True)

    @classmethod
    def main_show(cls):
        raise NotImplementedError


class VOC2007(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
                    'VOCtrainval_06-Nov-2007.tar'),
            'sha1': '34ed68851bce2a36e2a223fa52c661d592c66b3c',
        },
        'test': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
                    'VOCtest_06-Nov-2007.tar'),
            'sha1': '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1',
        },
    }


class VOC2008(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2008/'
                    'VOCtrainval_14-Jul-2008.tar'),
            'sha1': 'fc87d2477a1ae78c6748dc25b88c052eb8b06d75',
        },
        'test': {
            'url': ('https://s3.amazonaws.com/scikit-data/pascal/'
                    'VOC2008test.tar'),
            'sha1': '2044e7c61c407ca1f085e2bff5f188c7f7df7f48',
        },
    }


class VOC2009(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2009/'
                    'VOCtrainval_11-May-2009.tar'),
            'sha1': '0bc2be22b76a9bcb744c0458c535f3a84f054bbc',
        },
        'test': {
            'url': ('https://s3.amazonaws.com/scikit-data/pascal/'
                    'VOC2009test.tar'),
            'sha1': 'e638975ae3faca04aabc3ddb577d13e04da60950',
        }
    }


class VOC2010(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2010/'
                    'VOCtrainval_03-May-2010.tar'),
            'sha1': 'bf9985e9f2b064752bf6bd654d89f017c76c395a',
        },
        'test': {
            'url': ('https://s3.amazonaws.com/scikit-data/pascal/'
                    'VOC2010test.tar'),
            'sha1': '8f426aee2cb0ed0e07b5fceb45eff6a38595abfb',
        }
    }


class VOC2011(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2011/'
                    'VOCtrainval_25-May-2011.tar'),
            'sha1': '71ceda5bc8ce4a6486f7996b0924eee265133895',
            # the following will fix the fact that a prefix dir has been added
            # to the archive
            'moves': [{'source': 'TrainVal/VOCdevkit',
                       'destination': 'VOCdevkit'}],
        },
        'test': {
            'url': ('https://s3.amazonaws.com/scikit-data/pascal/'
                    'VOC2011test.tar.gz'),
            'sha1': 'e988fa911f2199309f76a6f44691e9471a011c45',
        }
    }


class VOC2012(BasePASCAL):
    ARCHIVES = {
        'trainval': {
            'url': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
                    'VOCtrainval_11-May-2012.tar'),
            'sha1': '4e443f8a2eca6b1dac8a6c57641b67dd40621a49',
            # the following will fix the fact that a prefix dir has been added
            # to the archive
            #'moves': [{'source': 'TrainVal/VOCdevkit',
            #           'destination': 'VOCdevkit'}],
        }
    }


def main_fetch():
    raise NotImplementedError


def main_show():
    raise NotImplementedError
