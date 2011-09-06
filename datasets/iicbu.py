# -*- coding: utf-8 -*-
"""IICBU 2008 Datasets

http://ome.grc.nia.nih.gov/iicbu2008

For more information, please refer to:

IICBU 2008 - A Proposed Benchmark Suite for Biological Image Analysis
Shamir, L., Orlov, N., Eckley, D.M., Macura, T., Goldberg, I.G.
Medical & Biological Engineering & Computing (2008)
Vol. 46, No. 9, pp. 943-947
http://ome.grc.nia.nih.gov/iicbu2008/IICBU2008-benchmark.pdf
"""

# Copyright (C) 2011
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>

# License: Simplified BSD

# XXX: standard categorization tasks (csv-based)


import os
from os import path
import shutil
from glob import glob
import hashlib

from data_home import get_data_home
from utils import download, extract


class BaseIICBU(object):
    """IICBU Biomedical Dataset

    Attributes
    ----------
    meta: list of dict
        Metadata associated with the dataset. For each image with index i,
        meta[i] is a dict with keys:
            name: str
                Name of the individual's face in the image.
            filename: str
                Full path to the image.
            id: int
                Identifier of the image.
            sha1: str
                SHA-1 hash of the image.

    Notes
    -----
    If joblib is available, then `meta` will be cached for faster
    processing. To install joblib use 'pip install -U joblib' or
    'easy_install -U joblib'.
    """

    EXTRACT_DIR = 'images'

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
        return path.join(get_data_home(), 'iicbu', self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""

        home = self.home()

        if not download_if_missing:
            raise IOError("'%s' exists!" % home)

        # download archive
        url = self.URL
        sha1 = self.SHA1
        basename = path.basename(url)
        archive_filename = path.join(home, basename)
        if not path.exists(archive_filename):
            if not download_if_missing:
                return
            if not path.exists(home):
                os.makedirs(home)
            download(url, archive_filename, sha1=sha1)

        # extract it
        dst_dirname = self.home(self.EXTRACT_DIR)
        if not path.exists(dst_dirname):
            extract(archive_filename, dst_dirname, sha1=sha1, verbose=True)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if hasattr(self, '_meta'):
            return self._meta
        else:
            self.fetch(download_if_missing=True)
            self._meta = self._get_meta()
            return self._meta

    def _get_meta(self):

        names = sorted(os.listdir(self.home(self.EXTRACT_DIR)))

        meta = []
        ind = 0

        for name in names:

            pattern = path.join(self.home(self.EXTRACT_DIR, name), '*.*')
            img_filenames = sorted(glob(pattern))

            for img_filename in img_filenames:
                img_data = open(img_filename, 'rb').read()
                sha1 = hashlib.sha1(img_data).hexdigest()

                data = dict(name=name,
                            id=ind,
                            filename=img_filename,
                            sha1=sha1)

                meta += [data]
                ind += 1

        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Standard Tasks
    # ------------------------------------------------------------------------
    # TODO


class Pollen(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/pollen.tar.gz'
    SHA1 = '99014ff40054b244b98474cd26125c55a90e0970'


class RNAi(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/rnai.tar.gz'
    SHA1 = '8de7f55c9a73b8d5050c8bc06f962de1d5a236ef'


class CelegansMuscleAge(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/celegans.tar.gz'
    SHA1 = '244404cb9504d39f765d2bf161a1ba32809e7256'


class TerminalBulbAging(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/terminalbulb.tar.gz'
    SHA1 = '2e81b3a5dea4df6c4e7d31f2999655084e54385b'


class Binucleate(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/binucleate.tar.gz'
    SHA1 = '7c0752899519b286c0948eb145fb2b6bd2bd2134'


class Lymphoma(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/lymphoma.tar.gz'
    SHA1 = '5af6bf000a9f7d0bb9b54ae0558fbeccc1758fe6'

#http://ome.grc.nia.nih.gov/iicbu2008/agemap/index.html'
#class LiverGenderCR(BaseIICBU):
#class LiverGenderAL(BaseIICBU):
#class LiverAging(BaseIICBU):

class Hela2D(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/hela.tar.gz'
    SHA1 = 'f5b13a8efd19dee9c53ab8da5ea6c017fdfb65a2'


class CHO(BaseIICBU):
    URL = 'http://ome.grc.nia.nih.gov/iicbu2008/cho.tar.gz'
    SHA1 = '0c55f49d34f50ef0a0d526afde0fa16fee07ba08'
