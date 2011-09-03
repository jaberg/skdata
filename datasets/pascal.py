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

from data_home import get_data_home
from utils import download, extract

import logging
logger = logging.getLogger(__name__)


class BasePASCAL(object):

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta

    @property
    def home(self):
        return path.join(get_data_home(), 'pascal', self.__class__.__name__)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self, download_if_missing=True):

        home = self.home
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
            self.fetch(download_if_missing=True)
            self._meta = self._build_meta()
            return self._meta

    def _build_meta(self):
        return "TODO"

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

voc07 = VOC2007()
print voc07.meta
