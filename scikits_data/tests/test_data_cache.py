import os
import shutil
import sys
import tempfile
import unittest

from scikits_data import data_cache

class TestCacheDir(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = data_cache.SCIKITS_DATA
        data_cache.SCIKITS_DATA = self.tmpdir

    def tearDown(self):
        data_cache.SCIKITS_DATA = self.orig
        shutil.rmtree(self.tmpdir)

    def test_simple(self):
        # at first foo does not exist
        foo = data_cache.get_cache_dir('foo')
        assert foo == os.path.join(data_cache.SCIKITS_DATA, 'foo')
        assert not os.listdir(foo)

        # now foo already exists
        foo2 = data_cache.get_cache_dir('foo')
        assert foo2 == foo

        barfoo = data_cache.get_cache_dir('bar/asdf/fooo')
        assert not os.listdir(barfoo)

        barfoo2 = data_cache.get_cache_dir('bar/asdf/fooo')
        assert not os.listdir(barfoo2)
        assert barfoo2 == barfoo
