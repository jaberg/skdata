import numpy as np

from skdata import caltech
from skdata import tasks

counts_101 = [467, 435, 435, 200, 798, 55, 800, 42, 42, 47, 54, 46, 33, 128,
98, 43, 85, 91, 50, 43, 123, 47, 59, 62, 107, 47, 69, 73, 70, 50, 51, 57, 67,
52, 65, 68, 75, 64, 53, 64, 85, 67, 67, 45, 34, 34, 51, 99, 100, 42, 54, 88,
80, 31, 64, 86, 114, 61, 81, 78, 41, 66, 43, 40, 87, 32, 76, 55, 35, 39, 47,
38, 45, 53, 34, 57, 82, 59, 49, 40, 63, 39, 84, 57, 35, 64, 45, 86, 59, 64, 35,
85, 49, 86, 75, 239, 37, 59, 34, 56, 39, 60]

caltech_101_breaks = [466, 901, 1336, 1536, 2334, 2389, 3189, 3231, 3273, 3320,
3374, 3420, 3453, 3581, 3679, 3722, 3807, 3898, 3948, 3991, 4114, 4161, 4220,
4282, 4389, 4436, 4505, 4578, 4648, 4698, 4749, 4806, 4873, 4925, 4990, 5058,
5133, 5197, 5250, 5314, 5399, 5466, 5533, 5578, 5612, 5646, 5697, 5796, 5896,
5938, 5992, 6080, 6160, 6191, 6255, 6341, 6455, 6516, 6597, 6675, 6716, 6782,
6825, 6865, 6952, 6984, 7060, 7115, 7150, 7189, 7236, 7274, 7319, 7372, 7406,
7463, 7545, 7604, 7653, 7693, 7756, 7795, 7879, 7936, 7971, 8035, 8080, 8166,
8225, 8289, 8324, 8409, 8458, 8544, 8619, 8858, 8895, 8954, 8988, 9044, 9083]


def test_Caltech101():
    dset = caltech.Caltech101()
    task = dset.img_classification_task(dtype='float32')
    tasks.assert_img_classification(*task, N=9144)

    X, y = task
    assert X[0].shape == (144, 145, 3)
    assert X[1].shape == (817, 656, 3)
    assert X[100].shape == (502, 388, 3)

    assert len(np.unique(y)) == 102  # number of categories
    ylist = y.tolist()
    counts = [ylist.count(z) for z in np.unique(ylist)]
    assert counts == counts_101

    z = y.copy()
    z.sort()
    assert (y == z).all()

    assert (y[1:] != y[:-1]).nonzero()[0].tolist() == caltech_101_breaks

def test_Caltech256():
    dset = caltech.Caltech256()
    task = dset.img_classification_task(dtype='float32')
    tasks.assert_img_classification(*task)
