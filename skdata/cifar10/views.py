import logging

import numpy as np

from .dataset import CIFAR10
from ..dslang import Task, BestModel, Score

logger = logging.getLogger(__name__)


class OfficialImageClassificationTask(object):
    def __init__(self, x_dtype='uint8', y_dtype='int', n_train=50000):
        if x_dtype not in ('uint8', 'float32'):
            raise TypeError()

        if y_dtype not in ('str', 'int'):
            raise TypeError()

        if not (0 <= n_train <= 50000):
            raise ValueError('n_train must fall in range(50000)', n_train)

        dataset = CIFAR10()
        dataset.meta  #trigger loading things

        y = dataset._labels
        if y_dtype == 'str':
            y = np.asarray(dataset.LABELS)[y]

        train = Task('image_classification',
                x=dataset._pixels[:n_train].astype(x_dtype),
                y=y[:n_train])
        test = Task('image_classification',
                x=dataset._pixels[50000:].astype(x_dtype),
                y=y[50000:])

        if 'float' in x_dtype:
            # N.B. that (a) _pixels are not writeable
            #      _pixels are uint8, so we must have copied
            train.x /= 255.0
            test.x /= 255.0

        self.dataset = dataset
        self.protocol = Score(BestModel(train), test)
        self.train = train
        self.test = test

class OfficialVectorClassificationTask(OfficialImageClassificationTask):
    def __init__(self, x_dtype='float32', y_dtype='int', n_train=50000):
        OfficialImageClassificationTask.__init__(self,
                x_dtype, y_dtype, n_train)
        self.train.x.shape = (len(self.train.x), 32 * 32 * 3)
        self.test.x.shape = (len(self.test.x), 32 * 32 * 3)


OfficialImageClassification = OfficialImageClassificationTask
OfficialVectorClassification = OfficialVectorClassificationTask


class StratifiedImageClassification(object):
    def __init__(self, dtype, n_train, n_valid, n_test, shuffle_seed=123,
            channel_major=False):

        if str(dtype) != 'uint8':
            raise NotImplementedError(dtype)

        self.n_classes = n_classes = 10
        assert n_train + n_valid <= 50000
        assert n_test <= 10000

        cf10 = CIFAR10()
        cf10.meta  # -- trigger data load

        # -- divide up the dataset as it was meant: train / test
        trn_images = cf10._pixels[:50000]
        trn_labels = cf10._labels[:50000]
        tst_images = cf10._pixels[50000:]
        tst_labels = cf10._labels[50000:]

        assert str(cf10._pixels.dtype) == 'uint8'

        # -- now carve it up so that we have balanced classes for fitting and
        #    validation
        logger.debug('re-indexing dataset')

        train = {}
        test = {}
        label_set = range(10)
        for label in label_set:
            train[label] = trn_images[trn_labels == label]
            test[label] = tst_images[tst_labels == label]
            assert len(train[label]) == len(trn_labels) / n_classes
            assert len(test[label]) == len(tst_labels) / n_classes

        del trn_images, trn_labels
        del tst_images, tst_labels

        if np.any(np.asarray([n_train, n_valid, n_test]) % len(label_set)):
            raise NotImplementedError('size not muptiple of 10',
                    (n_train, n_valid, n_test))
        else:
            trn_K = n_train // len(label_set)
            val_K = n_valid // len(label_set)
            tst_K = n_test // len(label_set)
            trn_images = np.concatenate([train[label][:trn_K]
                for label in label_set])
            trn_labels = np.concatenate([[label] * trn_K
                for label in label_set])

            assert len(trn_images) == len(trn_labels)
            assert trn_images.shape == (n_train, 32, 32, 3)
            assert trn_labels.shape == (n_train,)

            val_images = np.concatenate([train[label][trn_K:trn_K + val_K]
                for label in label_set])
            val_labels = np.concatenate([[label] * val_K
                for label in label_set])

            assert len(val_images) == len(val_labels)
            assert val_images.shape == (n_valid, 32, 32, 3)
            assert val_labels.shape == (n_valid,)

            tst_images = np.concatenate([test[label][:tst_K]
                for label in label_set])
            tst_labels = np.concatenate([[label] * tst_K
                for label in label_set])

            assert len(tst_images) == len(tst_labels)
            assert tst_images.shape == (n_test, 32, 32, 3)
            assert tst_labels.shape == (n_test,)

        logger.debug('done re-indexing dataset')
        def shuffle(X, s):
            if shuffle_seed:
                np.random.RandomState(shuffle_seed + s).shuffle(X)

            # -- hack to put it here, but it works for now
            if X.ndim > 1 and channel_major:
                X = X.transpose(0, 3, 1, 2).copy()

            return X

        self.dataset = cf10
        self.trn_images = shuffle(trn_images, 0)
        self.trn_labels = shuffle(trn_labels, 0)
        self.val_images = shuffle(val_images, 1)
        self.val_labels = shuffle(val_labels, 1)
        self.tst_images = shuffle(tst_images, 2)
        self.tst_labels = shuffle(tst_labels, 2)


        for images in self.trn_images, self.val_images, self.tst_images:
            assert str(images.dtype) == dtype, (images.dtype, dtype)

    def protocol(self, algo):

        # XXX: task should be idx, not images

        task_trn = Task(
            'image_classification',
            name='trn',
            x=self.trn_images,
            y=self.trn_labels,
            n_classes=self.n_classes)

        task_val = Task(
            'image_classification',
            name='val',
            x=self.val_images,
            y=self.val_labels,
            n_classes=self.n_classes)

        task_tst = Task(
            'image_classification',
            name='tst',
            x=self.tst_images,
            y=self.tst_labels,
            n_classes=self.n_classes)

        model = algo.best_model(train=task_trn, valid=task_val)

        algo.loss(model, task_tst)


