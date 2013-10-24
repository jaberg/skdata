"""
Specific learning problems for the kaggle_facial_expression dataset.

"""
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit

from ..dslang import Task

from dataset import KaggleFacialExpression


class ContestCrossValid(object):
    """
    This View implements the official contest evaluation protocol.
    
    https://www.kaggle.com/c/challenges-in-representation-learning-facial
    -expression-recognition-challenge

    """
    max_n_train = KaggleFacialExpression.N_TRAIN
    max_n_test = KaggleFacialExpression.N_TEST

    def __init__(self, x_dtype=np.float32,
                 n_train=max_n_train,
                 n_valid=0,
                 n_test=max_n_test,
                 shuffle_seed=123,
                 channel_major=False,
                 ds=None
                ):

        if ds is None:
            ds = KaggleFacialExpression()

        assert n_train + n_valid <= self.max_n_train
        assert n_test <= self.max_n_test

        if shuffle_seed:
            rng = np.random.RandomState(shuffle_seed)
        else:
            rng = None

        # examples x rows x cols
        all_pixels = np.asarray([mi['pixels'] for mi in ds.meta])
        all_labels = np.asarray([mi['label'] for mi in ds.meta],
                                dtype='int32')

        assert len(all_pixels) == self.max_n_test + self.max_n_train

        if channel_major:
            all_images = all_pixels[:, None, :, :].astype(x_dtype)
        else:
            all_images = all_pixels[:, :, :, None].astype(x_dtype)

        if 'float' in str(x_dtype):
            all_images /= 255

        for ii in xrange(self.max_n_train):
            assert ds.meta[ii]['usage'] == 'Training'

        if n_train < self.max_n_train:
            ((fit_idxs, val_idxs),) = StratifiedShuffleSplit(
                y=all_labels[:self.max_n_train],
                n_iterations=1,
                test_size=n_valid,
                train_size=n_train,
                indices=True,
                random_state=rng)
        else:
            fit_idxs = np.arange(self.max_n_train)
            val_idxs = np.arange(0)

        sel_idxs = np.concatenate([fit_idxs, val_idxs])
        if n_test < self.max_n_test:
            ((ign_idxs, tst_idxs),) = StratifiedShuffleSplit(
                y=all_labels[self.max_n_train:],
                n_iterations=1,
                test_size=n_test,
                indices=True,
                random_state=rng)
            tst_idxs += self.max_n_train
            del ign_idxs
        else:
            tst_idxs = np.arange(self.max_n_train, len(all_labels))

        self.dataset = ds
        self.n_classes = 7
        self.fit_idxs = fit_idxs
        self.val_idxs = val_idxs
        self.sel_idxs = sel_idxs
        self.tst_idxs = tst_idxs
        self.all_labels = all_labels
        self.all_images = all_images

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
                all_images=self.all_images,
                all_labels=self.all_labels,
                n_classes=self.n_classes)

        task_fit = task('fit', self.fit_idxs)
        task_val = task('val', self.val_idxs)
        task_sel = task('sel', self.sel_idxs)
        task_tst = task('tst', self.tst_idxs)

        if len(self.val_idxs):
            model1 = algo.best_model(train=task_fit, valid=task_val)
            yield ('model validation complete', model1)

        model2 = algo.best_model(train=task_sel)
        algo.loss(model2, task_tst)
        yield ('model testing complete', model2)

