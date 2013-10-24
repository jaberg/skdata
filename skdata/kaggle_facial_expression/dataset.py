"""
This data set was released as the

"Challengest in Representation Learning:
Facial Expression Recognition Challenge"

on Kaggle on April 12 2013, as part of an ICML-2013 workshop on representation
learning.

Kaggle Contest Description Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data consists of 48x48 pixel grayscale images of faces. The faces have
been automatically registered so that the face is more or less centered and
occupies about the same amount of space in each image. The task is to
categorize each face based on the emotion shown in the facial expression in to
one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad,
5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column
contains a numeric code ranging from 0 to 6, inclusive, for the emotion that
is present in the image. The "pixels" column contains a string surrounded in
quotes for each image. The contents of this string a space-separated pixel
values in row major order. test.csv contains only the "pixels" column and your
task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the
leaderboard consists of 3,589 examples. The final test set, to be released 72
hours before the end of the competition, consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part
of an ongoing research project. They have graciously provided the workshop
organizers with a preliminary version of their dataset to use for this
contest.

"""

import cPickle
import lockfile
import logging
import os
import shutil

import numpy as np

from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import verify_sha1, extract

logger = logging.getLogger(__name__)

FILES_SHA1s = [
    ('test.csv', '3f9199ae6e9a40137e72cb63264490e622d9c798'),
    ('train.csv', '97651a1fffc7e0af22ebdd5de36700b0e7e0c12c'),
    ('example_submission.csv', '14fac7ef24e3ab6d9fcaa9edd273fe08abfa5c51')]

TGZ_FILENAME = 'fer2013.tar.gz'

FULL_URL = '/'.join([
    'http://www.kaggle.com',
    'c',
    ('challenges-in-representation-learning'
     '-facial-expression-recognition-challenge'),
    'download',
    TGZ_FILENAME])

TGZ_SHA1 = 'b0e7632c70853f4d3b6a2a73031dd8c71d8d536d'


class KaggleFacialExpression(object):
    N_TRAIN = 28709
    N_TEST = 7178

    def __init__(self):
        self.name = 'kaggle_facial_expression'

    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def install(self, local_fer2013):
        """
        Verify SHA1 and copy given file into .skdata cache directory.
        """
        verify_sha1(local_fer2013, TGZ_SHA1)
        if not os.path.isdir(self.home()):
            os.makedirs(self.home())
        lock = lockfile.FileLock(self.home())
        if lock.is_locked():
            logger.warn('%s is locked, waiting for release' % self.home())

        with lock:
            shutil.copyfile(local_fer2013, self.home(TGZ_FILENAME))
            extract(self.home(TGZ_FILENAME), self.home())

    @property
    def meta(self):
        try:
            return self._meta
        except AttributeError:
            self._meta = self._get_meta()
            return self._meta

    def _get_meta(self):
        filename = 'meta_%s.pkl' % self._build_meta_version
        try:
            if self._build_meta_version:
                meta = cPickle.load(open(self.home(filename)))
            else:
                raise IOError()
        except (IOError, cPickle.PickleError), e:
            meta = self._build_meta()
            outfile = open(self.home(filename), 'wb')
            cPickle.dump(meta, outfile, -1)
            outfile.close()
        return meta

    _build_meta_version = '3'
    def _build_meta(self):
        meta = []

        # -- load train.csv
        for ii, line in enumerate(open(self.home('fer2013', 'fer2013.csv'))):
            if ii == 0:
                continue
            label, pixels, usage = line.split(',')
            assert int(label) < 7
            if 0:
                assert pixels[-3] == '"'
                assert pixels[0] == '"'
                pixels = np.asarray(map(int, pixels[1:-3].split(' ')), dtype=np.uint8)
            else:
                pixels = np.asarray(map(int, pixels.split(' ')), dtype=np.uint8)
            meta.append({
                'label': int(label),
                'pixels': pixels.reshape(48, 48),
                'usage': usage.strip(' \n'),
                })

        return meta


