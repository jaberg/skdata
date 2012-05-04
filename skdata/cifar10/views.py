from dataset import CIFAR10

class Split(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test

class Task(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

# XXX: Ugliness - why have to prefix all class names with
#                 ImageClassificationView{1,2}{Official,xxx}


# XXX protocol / view1 vs view2 / official vs. not /

class ImageClassificationView2Official(object):
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

        # XXX: x is big enough that not lazy-loading is an issue.

        x = self._pixels.astype(x_dtype)
        if 'float' in x_dtype:
            x = x / 255.0  # N.B. do a copy!

        train = Task(x[:50000], y[:50000])
        test = Task(x[50000:], y[50000:])

        split = Split(train, test)

        self.dataset = dataset
        self.splits = [split]


# XXX: Ugliness - why have to prefix all class names with
#                 ImageClassificationView{1,2}

class ImageClassificationView1RandomKfold(object):
    """
    Non-official stratified K-fold shuffling of the training data.
    """
    def __init__(self, x_dtype='uint8', y_dtype='int', rseed=1, K=5):
        if x_dtype not in ('uint8', 'float32'):
            raise TypeError()

        if y_dtype not in ('str', 'int'):
            raise TypeError()

        if y_dtype == 'str':
            raise NotImplementedError()

        dataset = CIFAR10()
        meta = dataset.meta  #trigger loading things

        y = dataset._labels

        idx_lists = sklearn.randomkfoldsplit(K=K, rng=np.RandomState(rseed),
                labels = dataset._labels)

        tasks = []
        assert len(idx_lists) == K
        for k in idx_list in enumerate(idx_lists):
            x = ltake(x
            task = Task(x=x, y=y)
            tasks.append(task)


