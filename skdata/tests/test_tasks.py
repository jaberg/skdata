import unittest
import numpy as np
from skdata import larochelle_etal_2007, tasks

def rnd(dtype, *shp):
    return np.random.rand(*shp).astype(dtype)

class TestAssertMethods(unittest.TestCase):
    def test_assert_classification(self):
        # things that work:
        tasks.assert_classification(
                rnd('float32', 4, 2), rnd('int8', 4))
        tasks.assert_classification(
                rnd('float64', 4, 2), rnd('uint64', 4))
        tasks.assert_classification(
                rnd('float64', 4, 2), rnd('uint64', 4), 4)

        # things that break:
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('int8', 4, 2), rnd('int8', 4))        # X not float
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float32', 4, 2), rnd('float64', 4))  # y not int
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float32', 4, 2), rnd('int8', 5))     # y wrong len
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float32', 4, 2), rnd('int8', 4, 1))  # y wrong rank
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float32', 4, 2), rnd('int8', 4, 7))  # y wrong rank
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float32', 4, 2, 2), rnd('int8', 4))  # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float64', 4), rnd('int8', 4))        # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_classification,
                rnd('float64', 4, 3), rnd('int8', 4), 5)  # N mismatch

    # TODO: test_assert_classification_train_valid_test

    def test_assert_regression(self):
        # things that work:
        tasks.assert_regression(
                rnd('float32', 4, 2), rnd('float64', 4, 1))
        tasks.assert_regression(
                rnd('float64', 4, 2), rnd('float32', 4, 3))
        tasks.assert_regression(
                rnd('float64', 4, 2), rnd('float32', 4, 3), 4)

        # things that break:
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('int8', 4, 2), rnd('float32', 4, 1))        # X not float
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float32', 4, 2), rnd('int32', 4, 1))       # y not float
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float32', 4, 2), rnd('float32', 5, 1))     # y wrong len
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float32', 4, 2), rnd('float32', 4))        # y wrong rank
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float32', 4, 2), rnd('float32', 4, 7, 3))  # y wrong rank
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float32', 4, 2, 2), rnd('float32', 4, 1))  # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float64', 4), rnd('float32', 4, 1))        # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_regression,
                rnd('float64', 4, 3), rnd('float32', 4, 1), 5)  # N mismatch


    # TODO: test_assert_matrix_completion

    def test_assert_latent_structure(self):
        # things that work:
        tasks.assert_latent_structure(rnd('float32', 4, 2))
        tasks.assert_latent_structure(rnd('float64', 11, 1))
        tasks.assert_latent_structure(rnd('float64', 11, 1), 11)

        # things that break:
        self.assertRaises(AssertionError, tasks.assert_latent_structure,
                rnd('int8', 4, 2))        # X not float
        self.assertRaises(AssertionError, tasks.assert_latent_structure,
                rnd('float32', 4, 2, 2))  # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_latent_structure,
                rnd('float64', 4))        # X wrong rank
        self.assertRaises(AssertionError, tasks.assert_latent_structure,
                rnd('float64', 4, 3), 5)  # N mismatch

    def test_classification_train_valid_test(self):

        dataset = larochelle_etal_2007.Rectangles() # smallest one with splits
        assert not hasattr(dataset, 'classification_train_valid_test_task')

        train, valid, test = tasks.classification_train_valid_test(dataset)
        tasks.assert_classification(*train)
        tasks.assert_classification(*valid)
        tasks.assert_classification(*test)

        assert len(train[0]) == dataset.descr['n_train']
        assert len(valid[0]) == dataset.descr['n_valid']
        assert len(test[0]) == dataset.descr['n_test']

        tasks.assert_classification_train_valid_test(train, valid, test)
