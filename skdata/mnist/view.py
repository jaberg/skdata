import numpy as np

from .dataset import MNIST
from ..dslang import Task

class OfficialImageClassification(object):
    def __init__(self, x_dtype='uint8', y_dtype='int'):
        self.dataset = dataset = MNIST()
        dataset.meta  # -- trigger load if necessary

        all_images = np.vstack((dataset.arrays['train_images'],
                                dataset.arrays['test_images']))
        all_labels = np.concatenate([dataset.arrays['train_labels'],
                                     dataset.arrays['test_labels']])

        if len(all_images) != 70000:
            raise ValueError()
        if len(all_labels) != 70000:
            raise ValueError()

        # TODO: add random shuffling options like in cifar10
        # XXX: ensure this is read-only view
        self.sel_idxs = np.arange(60000)
        self.tst_idxs = np.arange(60000, 70000)
        self.fit_idxs = np.arange(50000)
        self.val_idxs = np.arange(50000, 60000)

        # XXX: ensure this is read-only view
        self.all_images = all_images[:, :, :, np.newaxis].astype(x_dtype)
        self.all_labels = all_labels.astype(y_dtype)

        self.n_classes = 10

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


        model1 = algo.best_model(train=task_fit, valid=task_val)
        yield ('model validation complete', model1)

        model2 = algo.best_model(train=task_sel)
        algo.loss(model2, task_tst)
        yield ('model testing complete', model2)


class OfficialVectorClassification(OfficialImageClassification):
    def __init__(self, *args, **kwargs):
        OfficialImageClassification.__init__(self, *args, **kwargs)
        self.all_vectors = self.all_images.reshape(len(self.all_images), -1)

    def protocol_iter(self, algo):

        def task(name, idxs):
            return Task(
                'indexed_vector_classification',
                name=name,
                idxs=idxs,
                all_vectors=self.all_vectors,
                all_labels=self.all_labels,
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
