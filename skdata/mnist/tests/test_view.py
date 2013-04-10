import numpy as np
from skdata.mnist import view
from skdata.base import SklearnClassifier

def test_image_classification():

    class ConstantPredictor(object):
        def __init__(self, dtype='int', output=0):
            self.dtype = dtype
            self.output = output

        def fit(self, X, y):
            assert X.shape[0] in (50000, 60000), X.shape
            # -- the SklearnClassifier flattens image
            #    data sets to be compatible with sklearn classifiers,
            #    which expect vectors. That's why this method sees
            #    flattened images.
            assert np.prod(X.shape[1:]) == 784, X.shape

        def predict(self, X):
            return np.zeros(len(X), dtype=self.dtype) + self.output

    view_protocol = view.OfficialImageClassification()
    learn_algo = SklearnClassifier(ConstantPredictor)
    view_protocol.protocol(learn_algo)
    assert learn_algo.results['loss'][0]['task_name'] == 'tst'
    assert np.allclose(
            learn_algo.results['loss'][0]['err_rate'],
            0.9,
            atol=.01)


def test_vector_classification():

    class ConstantPredictor(object):
        def __init__(self, dtype='int', output=0):
            self.dtype = dtype
            self.output = output

        def fit(self, X, y):
            assert X.shape[0] in (50000, 60000), X.shape
            assert X.shape[1:] == (784,), X.shape

        def predict(self, X):
            return np.zeros(len(X), dtype=self.dtype) + self.output

    view_protocol = view.OfficialVectorClassification()
    learn_algo = SklearnClassifier(ConstantPredictor)
    view_protocol.protocol(learn_algo)
    assert learn_algo.results['loss'][0]['task_name'] == 'tst'
    assert np.allclose(
            learn_algo.results['loss'][0]['err_rate'],
            0.9,
            atol=.01)
