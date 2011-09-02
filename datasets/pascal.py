"""PASCAL Visual Object Classes Datasets

    http://pascallin.ecs.soton.ac.uk/challenges/VOC
"""

# Copyright (c) 2011 Nicolas Pinto <pinto@rowland.harvard.edu>
# Copyright (c) 2011 Nicolas Poilver <poilvert@rowland.harvard.edu>
# License: Simplified BSD

#from os import listdir, makedirs, remove
import os
from os import path
import shutil
import sys

import logging
import numpy as np
import urllib
import tarfile

from data_home import get_data_home
import larray
import utils
#import utils.image
from utils import download, extract

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
        for set, basename in self.ARCHIVES.iteritems():
            url = path.join(self.BASE_URL, basename)
            archive_filename = path.join(home, basename)
            if not path.exists(archive_filename):
                download(url, archive_filename)

        # extract them
        if not path.exists(path.join(home, 'VOCdevkit')):
            for set, basename in self.ARCHIVES.iteritems():
                archive_filename = path.join(home, basename)
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
        pass

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

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
    BASE_URL = 'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007'
    ARCHIVES = {
        'trainval': 'VOCtrainval_06-Nov-2007.tar',
        'test': 'VOCtest_06-Nov-2007.tar'
    }


def main_fetch():
    raise NotImplementedError


def main_show():
    raise NotImplementedError

VOC2007().meta
