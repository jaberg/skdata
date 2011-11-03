from skdata import caltech
from skdata import tasks


def test_Caltech101():
    dset = caltech.Caltech101()
    task = dset.img_classification_task(dtype='float32')
    tasks.assert_img_classification(*task, N=9144)

    X, y = task
    assert X[0].shape == (144, 145, 3)
    assert X[1].shape == (817, 656, 3)
    assert X[100].shape == (502, 388, 3)


def test_Caltech256():
    dset = caltech.Caltech256()
    task = dset.img_classification_task(dtype='float32')
    tasks.assert_img_classification(*task)
