import dataset
from ..base import Task

class VectorXV(object):
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
                n_classes=ds.descr['n_classes'])

        start = n_train
        end = n_train + n_valid
        valid = Task('vector_classification',
                name='valid',
                x=ds._inputs[start:end].reshape((end-start), -1),
                y=ds._labels[start:end],
                n_classes=ds.descr['n_classes'])

        start = n_train + n_valid
        end = n_train + n_valid + n_test
        test = Task('vector_classification',
                name='test',
                x=ds._inputs[start:end].reshape((end-start), -1),
                y=ds._labels[start:end],
                n_classes=ds.descr['n_classes'])

        model = algo.best_model(train=train, valid=valid)
        algo.loss(model, train)
        algo.loss(model, valid)
        return algo.loss(model, test)


class MNIST_Basic_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Basic()


class MNIST_BackgroundImages_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_BackgroundImages()


class MNIST_BackgroundRandom_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_BackgroundRandom()


class MNIST_Rotated_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Rotated()


class MNIST_RotatedBackgroundImages_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_RotatedBackgroundImages()


class MNIST_Noise1_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise1()


class MNIST_Noise2_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise2()


class MNIST_Noise3_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise3()


class MNIST_Noise4_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise4()


class MNIST_Noise5_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise5()


class MNIST_Noise6_VectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.MNIST_Noise6()


class RectanglesVectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.Rectangles()


class RectanglesImagesVectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.RectanglesImages()


class ConvexVectorXV(VectorXV):
    def __init__(self):
        self.dataset = dataset.Convex()

