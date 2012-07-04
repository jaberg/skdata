
from .dataset import MNIST
import numpy as np
from numpy import newaxis

from ..dslang import Task, BestModel, Score

class OfficialImageClassification(object):
    def __init__(self, x_dtype='uint8', y_dtype='int'):
        self.dataset = dataset = MNIST()
        dataset.meta  # -- trigger load if necessary

        train = Task('image_classification',
                x=dataset.arrays['train_images']
                    [:, :, :, newaxis].astype(x_dtype),
                y=dataset.arrays['train_labels'].astype(y_dtype))

        test = Task('image_classification',
                x=dataset.arrays['test_images']
                    [:, :, :, newaxis].astype(x_dtype),
                y=dataset.arrays['test_labels'].astype(y_dtype))

        if str(x_dtype).startswith('float'):
            train.x = train.x / 255
            test.x = test.x / 255

        self.protocol = Score(BestModel(train), test)
        self.train = train
        self.test = test


class OfficialVectorClassification(OfficialImageClassification):
    def __init__(self, x_dtype='float32', y_dtype='int'):
        OfficialImageClassification.__init__(self, x_dtype, y_dtype)
        self.train.x.shape = (len(self.train.x), 28 * 28)
        self.test.x.shape = (len(self.test.x), 28 * 28)


