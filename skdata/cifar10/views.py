
from .dataset import CIFAR10
from ..dslang import Task, BestModel, Score


class OfficialImageClassificationTask(object):
    def __init__(self, x_dtype='uint8', y_dtype='int', n_train=50000):
        if x_dtype not in ('uint8', 'float32'):
            raise TypeError()

        if y_dtype not in ('str', 'int'):
            raise TypeError()

        if not (0 <= n_train <= 50000):
            raise ValueError('n_train must fall in range(50000)', n_train)

        dataset = CIFAR10()
        meta = dataset.meta  #trigger loading things

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

