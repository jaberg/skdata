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

from skdata.lfw import dataset, view
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


class EmptyLFW(dataset.BaseLFW):
    ARCHIVE_NAME = "i_dont_exist.tgz"
    img_shape = (250, 250, 3)
    COLOR = True
    IMAGE_SUBDIR = 'image_subdir'

    def home(self, *names):
        return os.path.join(SCIKIT_LEARN_DATA_EMPTY, 'lfw', self.name, *names)

    def fetch(self, download_if_missing=True):
        return


class FakeLFW(dataset.BaseLFW):
    IMAGE_SUBDIR = 'lfw_fake'  # corresponds to lfw, lfw_funneled, lfw_aligned
    img_shape = (250, 250, 3)
    COLOR = True

    def home(self, *names):
        return os.path.join(SCIKIT_LEARN_DATA, 'lfw', self.name, *names)

    def fetch(self, download_if_missing=True):
        if not os.path.exists(self.home()):
            os.makedirs(self.home())

        random_state = random.Random(42)
        np_rng = np.random.RandomState(42)

        # generate some random jpeg files for each person
        counts = FakeLFW.counts = {}
        for name in FAKE_NAMES:
            folder_name = self.home('images', self.IMAGE_SUBDIR, name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            n_faces = np_rng.randint(1, 5)
            counts[name] = n_faces
            for i in range(n_faces):
                file_path = os.path.join(folder_name,
                                         name + '_%04d.jpg' % (i + 1))
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
                    f.write('%s\t%d\t%d\n' % (name, first + 1, second + 1))

                for i in range(n_match):
                    first_name, second_name = random_state.sample(FAKE_NAMES, 2)
                    first_index = random_state.choice(range(counts[first_name]))
                    second_index = random_state.choice(range(counts[second_name]))
                    f.write('%s\t%d\t%s\t%d\n' % (first_name, first_index + 1,
                                                  second_name, second_index + 1))
            f.close()
        write_fake_pairs('pairsDevTrain.txt', 5, 1)
        write_fake_pairs('pairsDevTest.txt', 7, 1)
        write_fake_pairs('pairs.txt', 4, 3)


class FP_Empty(view.FullProtocol):
    DATASET_CLASS = EmptyLFW


class FP_Fake(view.FullProtocol):
    DATASET_CLASS = FakeLFW


def setup_module():
    """Test fixture run once and common to all tests of this module"""
    FakeLFW().fetch()


def teardown_module():
    """Test fixture (clean up) run once after all tests of this module"""
    FakeLFW().clean_up()


def test_empty_load():
    assert len(EmptyLFW().meta) == 0


def test_fake_load():
    fake = FakeLFW()
    counts_copy = dict(FakeLFW.counts)
    for m in fake.meta:
        counts_copy[m['name']] -= 1
    assert all(c == 0 for c in counts_copy.values())

    for m in fake.meta:
        assert m['filename'].endswith('.jpg')


def test_fake_verification_task():
    fake = FakeLFW()

    assert fake.pairsDevTrain.shape == (1, 2, 5, 2), fake.pairsDevTrain.shape
    assert fake.pairsDevTest.shape == (1, 2, 7, 2)
    assert fake.pairsView2.shape == (3, 2, 4, 2)

    for i in range(5):
        (lname, lnum) = fake.pairsDevTrain[0, 0, i, 0]
        (rname, rnum) = fake.pairsDevTrain[0, 0, i, 1]
        assert lname == rname

    for i in range(5):
        (lname, lnum) = fake.pairsDevTrain[0, 1, i, 0]
        (rname, rnum) = fake.pairsDevTrain[0, 1, i, 1]
        assert lname != rname



def test_fake_imgs():
    fp = FP_Fake()
    # test the default case
    images = fp.image_pixels
    assert images.dtype == 'uint8', images.dtype
    assert images.ndim == 4, images.ndim
    assert images.shape == (17, 250, 250, 3), images.shape

    img0 = images[0]
    assert isinstance(img0, np.ndarray)
    assert img0.dtype == 'uint8'
    assert img0.ndim == 3
    assert img0.shape == (250, 250, 3)


