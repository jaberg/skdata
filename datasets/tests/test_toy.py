
from datasets import toy

# XXX : this is  starting to look like a generic /task/ definition
def check_classification_Xy(X, y, N=None):
    A, B = X.shape
    C, = y.shape
    assert A == C
    if N is not None:
        assert A == N
    assert 'int' in str(y.dtype), y.dtype


def test_iris():
    iris = toy.Iris()
    assert len(iris.meta) == 150
    assert iris.meta[0]['sepal_length'] == 5.1
    assert iris.meta[-1]['petal_width'] == 1.8
    assert iris.meta[-2]['name'] == 'virginica'

    X, y = iris.classification_task()
    check_classification_Xy(X, y, len(iris.meta))
    assert y.min() == 0
    assert y.max() == 2


def test_digits():
    digits = toy.Digits()
    assert len(digits.meta) == 1797, len(digits.meta)
    assert digits.descr  #ensure it's been loaded
    assert digits.meta[3]['img'].shape == (8, 8)
    X, y = digits.classification_task()
    check_classification_Xy(X, y, len(digits.meta))
    assert y.min() == 0
    assert y.max() == 9


def test_diabetes():
    diabetes = toy.Diabetes()
    assert len(diabetes.meta) == 442, len(diabetes.meta)
    X, y = diabetes.classification_task()
    check_classification_Xy(X, y, len(diabetes.meta))
