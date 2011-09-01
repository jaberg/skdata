
from datasets import toy

def test_iris():
    iris = toy.Iris()
    assert len(iris.meta) == 150
    assert iris.meta[0]['sepal_length'] == 5.1
    assert iris.meta[-1]['petal_width'] == 1.8
    assert iris.meta[-2]['name'] == 'virginica'

    X, y = iris.classification_task()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert y.min() == 0 and y.max() == 2


def test_digits():
    digits = toy.Digits()
    assert len(digits.meta) == 1797
    assert digits.descr  #ensure it's been loaded
    assert digits.meta[3]['img'].shape == (8, 8)
    X, y = digits.classification_task()
    assert X.shape == (len(digits.meta), 64)
    assert y.min() == 0
    assert y.max() == 9
