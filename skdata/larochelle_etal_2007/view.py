import dataset
from ..base import Task

class RectanglesVectorXV(object):
    def __init__(self):
        self.dataset = dataset.Rectangles()

    def protocol(self, algo):
        ds = self.dataset
        ds.fetch(True)
        ds.build_meta()
        n_train = ds.descr['n_train']
        n_valid = ds.descr['n_valid']
        n_test = ds.descr['n_test']

        start = 0
        end = n_train
        train = Task('vector_classification',
                name='train',
                x=ds._inputs[start:end].reshape((end-start), -1),
                y=ds._labels[start:end],
                n_classes=2)

        start = n_train
        end = n_train + n_valid
        valid = Task('vector_classification',
                name='valid',
                x=ds._inputs[start:end].reshape((end-start), -1),
                y=ds._labels[start:end],
                n_classes=2)

        start = n_train + n_valid
        end = n_train + n_valid + n_test
        test = Task('vector_classification',
                name='test',
                x=ds._inputs[start:end].reshape((end-start), -1),
                y=ds._labels[start:end],
                n_classes=2)

        model = algo.best_model(train=train, valid=valid)
        algo.loss(model, train)
        algo.loss(model, valid)
        return algo.loss(model, test)


