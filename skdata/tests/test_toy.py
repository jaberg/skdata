
from skdata import toy
from skdata.iris import Iris
from skdata.diabetes import Diabetes
from skdata.digits import Digits
# TODO: move these datasets into their own file too
from skdata.toy import Linnerud
from skdata.toy import Boston
from skdata.toy import SampleImages
from skdata.tasks import assert_classification, assert_regression

def check_classification_Xy(X, y, N=None):
    assert_classification(X, y, N)


def check_regression_XY(X, Y, N=None):
    assert_regression(X, Y, N)


def test_iris():
    iris = Iris()
    assert len(iris.meta) == 150
    assert iris.meta[0]['sepal_length'] == 5.1
    assert iris.meta[-1]['petal_width'] == 1.8
    assert iris.meta[-2]['name'] == 'virginica'

    X, y = iris.classification_task()
    check_classification_Xy(X, y, len(iris.meta))
    assert y.min() == 0
    assert y.max() == 2


def test_digits():
    digits = Digits()
    assert len(digits.meta) == 1797, len(digits.meta)
    assert digits.descr  #ensure it's been loaded
    assert digits.meta[3]['img'].shape == (8, 8)
    X, y = digits.classification_task()
    check_classification_Xy(X, y, len(digits.meta))
    assert y.min() == 0
    assert y.max() == 9


def test_diabetes():
    diabetes = Diabetes()
    assert len(diabetes.meta) == 442, len(diabetes.meta)
    X, y = diabetes.classification_task()
    check_classification_Xy(X, y, len(diabetes.meta))


def test_linnerud():
    linnerud = Linnerud()
    assert len(linnerud.meta) == 20
    assert list(sorted(linnerud.meta[5].keys())) == [
            'chins', 'jumps', 'pulse', 'situps', 'waist', 'weight']
    X, Y = linnerud.regression_task()
    check_regression_XY(X, Y, 20)


def test_boston():
    boston = Boston()
    assert len(boston.meta) == 506
    keys = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD",
            "TAX","PTRATIO","B","LSTAT","MEDV"]
    assert set(keys) == set(boston.meta[6].keys())
    X, Y = boston.regression_task()
    check_regression_XY(X, Y, 506)


def test_sample_images():
    si = SampleImages()
    assert len(si.meta) == 2, len(si.meta)
    images = si.images()
    assert len(images) == 2
    assert images[0].shape == (427, 640, 3)
    assert images[1].shape == (427, 640, 3)
