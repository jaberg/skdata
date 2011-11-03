import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from skdata import utils

def test_random_spd_matrix():
    X = utils.random_spd_matrix(n_dim=5, random_state=0)

    assert_equal(X.shape, (5, 5), "X shape mismatch")
    assert_array_almost_equal(X, X.T)

    from numpy.linalg import eig
    eigenvalues, _ = eig(X)
    assert_equal(eigenvalues > 0, np.array([True] * 5),
                 "X is not positive-definite")
