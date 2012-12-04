import logging
from sklearn.cross_validation import StratifiedShuffleSplit

import numpy as np

from .dataset import CIFAR10
from ..dslang import Task

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
    """
    Data set is partitioned at top level into a testing set (tst) and a model
    selection set (sel).  The selection set is subdivided into a fitting set
    (fit) and a validation set (val).

    The evaluation protocol is to fit a classifier to the (fit) set, and judge
    it on (val). The best model on (val) is re-trained on the entire selection
    set, and finally evaluated on the test set.

    """
    def __init__(self, dtype, n_train, n_valid, n_test, shuffle_seed=123,
            channel_major=False):


        assert n_train + n_valid <= 50000
        assert n_test <= 10000

        cf10 = CIFAR10()
        cf10.meta  # -- trigger data load
        if str(dtype) != str(cf10._pixels.dtype):
            raise NotImplementedError(dtype)

        # -- divide up the dataset as it was meant: train / test

        if shuffle_seed:
            rng = np.random.RandomState(shuffle_seed)
        else:
            rng = None

        ((fit_idxs, val_idxs),) = StratifiedShuffleSplit(
            y=cf10._labels[:50000],
            n_iterations=1,
            test_size=n_valid,
            train_size=n_train,
            indices=True,
            random_state=rng)

        sel_idxs = np.concatenate([fit_idxs, val_idxs])

        if n_test < 10000:
            ((ign_idxs, tst_idxs),) = StratifiedShuffleSplit(
                y=cf10._labels[50000:],
                n_iterations=1,
                test_size=n_test,
                indices=True,
                random_state=rng)
            tst_idxs += 50000
            del ign_idxs
        else:
            tst_idxs = np.arange(50000, 60000)

        self.dataset = cf10
        self.n_classes = 10
        self.fit_idxs = fit_idxs
        self.val_idxs = val_idxs
        self.sel_idxs = sel_idxs
        self.tst_idxs = tst_idxs

    def protocol(self, algo):
        for _ in self.protocol_iter(algo):
            pass
        return algo

    def protocol_iter(self, algo):

        def task(name, idxs):
            return Task(
                'indexed_image_classification',
                name=name,
                idxs=idxs,
                all_images=self.dataset._pixels,
                all_labels=self.dataset._labels,
                n_classes=self.n_classes)

        task_fit = task('fit', self.fit_idxs)
        task_val = task('val', self.val_idxs)
        task_sel = task('sel', self.sel_idxs)
        task_tst = task('tst', self.tst_idxs)


        model1 = algo.best_model(train=task_fit, valid=task_val)
        yield ('model validation complete', model1)

        model2 = algo.best_model(train=task_sel)
        algo.loss(model2, task_tst)
        yield ('model testing complete', model2)


