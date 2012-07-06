
from .dataset import CIFAR10
from ..dslang import Task, BestModel, Score


class OfficialImageClassificationTask(object):
    def __init__(self, x_dtype='uint8', y_dtype='int'):
        if x_dtype not in ('uint8', 'float32'):
            raise TypeError()

        if y_dtype not in ('str', 'int'):
            raise TypeError()

        dataset = CIFAR10()
        meta = dataset.meta  #trigger loading things

        y = dataset._labels
        if y_dtype == 'str':
            y = np.asarray(dataset.LABELS)[y]

        x = dataset._pixels.astype(x_dtype)
        if 'float' in x_dtype:
            # N.B. working in-place could invalidate the original data set
            x = x / 255.0

        train = Task('image_classification', x=x[:50000], y=y[:50000])
        test = Task('image_classification', x=x[50000:], y=y[50000:])

        self.dataset = dataset
        self.protocol = Score(BestModel(train), test)
        self.train = train
        self.test = test


class OfficialVectorClassificationTask(OfficialImageClassificationTask):
    def __init__(self, x_dtype='float32', y_dtype='int'):
        OfficialImageClassificationTask.__init__(self, x_dtype, y_dtype)
        self.train.x.shape = (len(self.train.x), 32 * 32 * 3)
        self.test.x.shape = (len(self.test.x), 32 * 32 * 3)

