"""This test for the LFW require medium-size data dowloading and processing

If the data has not been already downloaded by runnning the examples,
the tests won't run (skipped).

If the test are run, the first execution will be long (typically a bit
more than a couple of minutes) but as the dataset loader is leveraging
joblib, successive runs will be fast (less than 200ms).
"""

import random
import os
import shutil
import tempfile
import numpy as np
try:
    try:
        from scipy.misc import imsave
    except ImportError:
        from scipy.misc.pilutil import imsave
except ImportError:
    imsave = None

from datasets.lfw import BaseLFW

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
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


class EmptyLFW(BaseLFW):
    NAME = 'Empty'
    ARCHIVE_NAME = "i_dont_exist.tgz"

    @classmethod
    def home(cls, *names):
        return os.path.join(SCIKIT_LEARN_DATA_EMPTY, 'lfw', cls.NAME, *names)


class FakeLFW(BaseLFW):
    NAME = 'Fake'
    IMAGEDIR = 'lfw_fake' #corresponds to lfw_funneled

    @classmethod
    def home(cls, *names):
        return os.path.join(SCIKIT_LEARN_DATA, 'lfw', cls.NAME, *names)

    @classmethod
    def fetch(cls, download_if_missing=True):
        if not os.path.exists(cls.home()):
            os.makedirs(cls.home())

        random_state = random.Random(42)
        np_rng = np.random.RandomState(42)

        # generate some random jpeg files for each person
        counts = cls.counts = {}
        for name in FAKE_NAMES:
            folder_name = cls.home(cls.IMAGEDIR, name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            n_faces = np_rng.randint(1, 5)
            counts[name] = n_faces
            for i in range(n_faces):
                file_path = os.path.join(folder_name, name + '_%04d.jpg' % i)
                uniface = np_rng.randint(0, 255, size=(250, 250, 3))
                try:
                    imsave(file_path, uniface)
                except ImportError:
                    # PIL is not properly installed, skip those tests
                    raise SkipTest

        if not os.path.exists(cls.home('lfw_funneled')):
            os.makedirs(cls.home('lfw_funneled'))

        # add some random file pollution to test robustness
        f = open(os.path.join(cls.home(), 'lfw_funneled', '.test.swp'), 'wb')
        f.write('Text file to be ignored by the dataset loader.')
        f.close()

        # generate some pairing metadata files using the same format as LFW
        f = open(os.path.join(cls.home(), 'pairsDevTrain.txt'), 'wb')
        f.write("10\n")
        more_than_two = [name for name, count in counts.iteritems()
                         if count >= 2]
        for i in range(5):
            name = random_state.choice(more_than_two)
            first, second = random_state.sample(range(counts[name]), 2)
            f.write('%s\t%d\t%d\n' % (name, first, second))

        for i in range(5):
            first_name, second_name = random_state.sample(FAKE_NAMES, 2)
            first_index = random_state.choice(range(counts[first_name]))
            second_index = random_state.choice(range(counts[second_name]))
            f.write('%s\t%d\t%s\t%d\n' % (first_name, first_index,
                                          second_name, second_index))
        f.close()

        f = open(os.path.join(cls.home(), 'pairsDevTest.txt'), 'wb')
        f.write("Fake place holder that won't be tested")
        f.close()

        f = open(os.path.join(cls.home(), 'pairs.txt'), 'wb')
        f.write("Fake place holder that won't be tested")
        f.close()


def setup_module():
    """Test fixture run once and common to all tests of this module"""
    FakeLFW.fetch()


def teardown_module():
    """Test fixture (clean up) run once after all tests of this module"""
    FakeLFW.erase()


@raises(IOError)
def test_load_empty_lfw_people():
    EmptyLFW.load_people()


@raises(IOError)
def test_load_empty_lfw_pairs():
    EmptyLFW.load_pairs_train()


def test_load_people_gets_everyone():
    people = FakeLFW.load_people()
    counts_copy = dict(FakeLFW.counts)
    for name in people.names:
        counts_copy[name] -= 1
    assert all(c == 0 for c in counts_copy.values())


def test_images_proxy():
    people = FakeLFW.load_people()
    images = people.images
    true_n_images = sum(FakeLFW.counts.values())

    assert images.dtype == 'uint8'
    assert images.ndim == 4
    assert images.shape == (true_n_images, 250, 250, 3)

    assert images[0].dtype == 'uint8'
    assert images[0].ndim == 3
    assert images[0].shape == (250, 250, 3)


def test_load_fake_lfw_pairs():
    pairs = FakeLFW.load_pairs_train()
    raise NotImplementedError()

