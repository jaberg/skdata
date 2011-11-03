from skdata import caltech
from skdata import tasks


def test_Caltech101():
    dset = caltech.Caltech101()
    task = dset.img_classification_task(dtype='float')
    tasks.assert_img_classification(*task)


def test_Caltech256():
    dset = caltech.Caltech256()
    task = dset.img_classification_task(dtype='float')
    tasks.assert_img_classification(*task)
