
from .dataset import CIFAR10
from ..dslang import Task, Split, BestModel, Score


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

        x = self._pixels.astype(x_dtype)
        if 'float' in x_dtype:
            # N.B. working in-place could invalidate the original data set
            x = x / 255.0

        train = Task('image_classification', x=x[:50000], y=y[:50000])
        test = Task('image_classification', x=x[50000:], y=y[50000:])

        split = Split(train, test)

        self.dataset = dataset
        self.splits = [split]

        self.protocol = Score(BestModel(train), test)

