"""This test for the LFW require medium-size data dowloading and processing

If the data has not been already downloaded by runnning the examples,
the tests won't run (skipped).

If the test are run, the first execution will be long (typically a bit
more than a couple of minutes) but as the dataset loader is leveraging
joblib, successive runs will be fast (less than 200ms).
"""

import random
import os
import tempfile
import numpy as np
try:
    try:
        from scipy.misc import imsave
    except ImportError:
        from scipy.misc.pilutil import imsave
except ImportError:
    imsave = None

from skdata import lfw
from skdata import tasks

from numpy.testing import assert_raises
from nose import SkipTest
from nose.tools import raises


SCIKIT_LEARN_DATA = tempfile.mkdtemp(prefix="scikit_learn_lfw_test_")
SCIKIT_LEARN_DATA_EMPTY = tempfile.mkdtemp(prefix="scikit_learn_empty_test_")

FAKE_NAMES = [
    'Abdelatif_Smith',
    'Abhati_Kepler',
    'Camara_Alvaro',
    'Chen_Dupont',
    'John_Lee',
    'Lin_Bauman',
    'Onur_Lopez',
]

def namelike(fullpath):
    """Returns part of an image full path that has the name in it"""
    return fullpath[-18:-8]


class EmptyLFW(lfw.BaseLFW):
    NAME = 'Empty'
    ARCHIVE_NAME = "i_dont_exist.tgz"

    def home(self, *names):
        return os.path.join(SCIKIT_LEARN_DATA_EMPTY, 'lfw', self.NAME, *names)

    def fetch(self, download_if_missing=True):
        return


class FakeLFW(lfw.BaseLFW):
    NAME = 'Fake'
    IMAGEDIR = 'lfw_fake' # corresponds to lfw, lfw_funneled, lfw_aligned

    def home(self, *names):
        return os.path.join(SCIKIT_LEARN_DATA, 'lfw', self.NAME, *names)

    def fetch(self, download_if_missing=True):
        if not os.path.exists(self.home()):
            os.makedirs(self.home())

        random_state = random.Random(42)
        np_rng = np.random.RandomState(42)

        # generate some random jpeg files for each person
        counts = FakeLFW.counts = {}
        for name in FAKE_NAMES:
            folder_name = self.home(self.IMAGEDIR, name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            n_faces = np_rng.randint(1, 5)
            counts[name] = n_faces
            for i in range(n_faces):
                file_path = os.path.join(folder_name, name + '_%04d.jpg' % (i+1))
                uniface = np_rng.randint(0, 255, size=(250, 250, 3))
                try:
                    imsave(file_path, uniface)
                except ImportError:
                    # PIL is not properly installed, skip those tests
                    raise SkipTest

        if not os.path.exists(self.home('lfw_funneled')):
            os.makedirs(self.home('lfw_funneled'))

        # add some random file pollution to test robustness
        f = open(os.path.join(self.home(), 'lfw_funneled', '.test.swp'), 'wb')
        f.write('Text file to be ignored by the dataset loader.')
        f.close()

        # generate some pairing metadata files using the same format as LFW
        def write_fake_pairs(filename, n_match, n_splits):
            f = open(os.path.join(self.home(), filename), 'wb')
            if n_splits == 1:
                f.write("%d\n" % n_match)
            else:
                f.write("%d\t%i\n" % (n_splits, n_match))
            for split in xrange(n_splits):
                more_than_two = [name for name, count in counts.iteritems()
                                 if count >= 2]
                for i in range(n_match):
                    name = random_state.choice(more_than_two)
                    first, second = random_state.sample(range(counts[name]), 2)
                    f.write('%s\t%d\t%d\n' % (name, first+1, second+1))

                for i in range(n_match):
                    first_name, second_name = random_state.sample(FAKE_NAMES, 2)
                    first_index = random_state.choice(range(counts[first_name]))
                    second_index = random_state.choice(range(counts[second_name]))
                    f.write('%s\t%d\t%s\t%d\n' % (first_name, first_index+1,
                                                  second_name, second_index+1))
            f.close()
        write_fake_pairs('pairsDevTrain.txt', 5, 1)
        write_fake_pairs('pairsDevTest.txt', 7, 1)
        write_fake_pairs('pairs.txt', 4, 3)


def setup_module():
    """Test fixture run once and common to all tests of this module"""
    FakeLFW().fetch()


def teardown_module():
    """Test fixture (clean up) run once after all tests of this module"""
    FakeLFW().clean_up()


@raises(IOError)
def test_empty_load():
    EmptyLFW().meta


def test_fake_load():
    fake = FakeLFW()
    counts_copy = dict(FakeLFW.counts)
    for m in fake.meta:
        counts_copy[m['name']] -= 1
    assert all(c == 0 for c in counts_copy.values())

    assert list(sorted(m['pairs'].keys())) == [
            'DevTest', 'DevTrain', 'fold_0', 'fold_1', 'fold_2']


def test_fake_classification_task():
    fake = FakeLFW()
    paths, labels = fake.raw_classification_task()
    
    assert len(paths) == len(labels)
    assert all(p.endswith('.jpg') for p in paths)
    assert 'int' in str(labels.dtype)

    #assert that names and labels correspond 1-1
    sig_l = {}
    sig_p = {}
    for p, l in zip(paths, labels):
        assert sig_l.setdefault(l, namelike(p)) == namelike(p)
        assert sig_p.setdefault(namelike(p), l) == l


def test_fake_verification_task():
    fake = FakeLFW()
    for split in (None, 'DevTrain', 'DevTest', 'fold_0', 'fold_1', 'fold_2'):
        if split is None:
            lpaths, rpaths, labels = fake.raw_verification_task()
        else:
            lpaths, rpaths, labels = fake.raw_verification_task(split=split)

        if split is None or split == 'DevTrain':
            assert len(labels) == 5 * 2
        elif split == 'DevTest':
            assert len(labels) == 7 * 2
        elif split.startswith('fold'):
            assert len(labels) == 4 * 2

        assert len(lpaths) == len(labels)
        assert len(rpaths) == len(labels)
        assert 'int' in str(labels.dtype)
        for l, r, t in zip(lpaths, rpaths, labels):
            assert t in (0, 1)
            if t == 0:
                assert namelike(l) != namelike(r)
            else:
                assert namelike(l) == namelike(r)

    assert_raises(KeyError, fake.raw_verification_task, split='invalid')


def test_fake_imgs():
    fake = FakeLFW()
    true_n_images = sum(fake.counts.values())
    # test the default case
    images, labels = fake.img_classification_task()
    assert images.dtype == 'uint8'
    assert images.ndim == 4
    assert images.shape == (true_n_images, 250, 250, 3)

    assert images[0].dtype == 'uint8'
    assert images[0].ndim == 3
    assert images[0].shape == (250, 250, 3)

    # test specified dtypes
    for dtype in 'uint8', 'float32':
        images, labels = fake.img_classification_task(dtype=dtype)
        assert images.dtype == dtype
        assert images.ndim == 4
        assert images.shape == (true_n_images, 250, 250, 3)

        assert images[0].dtype == dtype
        assert images[0].ndim == 3
        assert images[0].shape == (250, 250, 3)
        
        
def test_img_classification_task():
    dset = lfw.Original()
    X, y = dset.img_classification_task(dtype='float')
    tasks.assert_img_classification(X, y)
    
    
def test_img_verification_task():
    dset = lfw.Original()
    X, Y, z = dset.img_verification_task(dtype='float')
    tasks.assert_img_verification(X, Y, z)
