from skdata import caltech
from skdata import tasks


def test_Caltech101():
    dset = caltech.Caltech101()
    tasks.assert_img_classification(*dset.img_classification_task())


def test_Caltech256():
    dset = caltech.Caltech256()
    tasks.assert_img_classification(*dset.img_classification_task())
