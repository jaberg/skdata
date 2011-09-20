import numpy as np
from numpy.testing import assert_equal, assert_approx_equal, \
                          assert_array_almost_equal, assert_array_less

from skdata import synthetic as SG
from skdata import tasks

def test_madelon():
    madelon = SG.Madelon(n_samples=100, n_features=20, n_informative=5,
                               n_redundant=1, n_repeated=1, n_classes=3,
                               n_clusters_per_class=1, hypercube=False,
                               shift=None, scale=None, weights=[0.1, 0.25],
                               random_state=0)
    X, y = madelon.classification_task()
    tasks.assert_classification(X, y, 100)

    assert_equal(X.shape, (100, 20), "X shape mismatch")
    assert_equal(y.shape, (100,), "y shape mismatch")
    assert_equal(np.unique(y).shape, (3,), "Unexpected number of classes")
    assert_equal(sum(y == 0), 10, "Unexpected number of samples in class #0")
    assert_equal(sum(y == 1), 25, "Unexpected number of samples in class #1")
    assert_equal(sum(y == 2), 65, "Unexpected number of samples in class #2")


def test_four_regions():
    four_regions = SG.FourRegions(n_samples=100, random_state=0)
    X, y = four_regions.classification_task()
    tasks.assert_classification(X, y, 100)

    assert_equal(X.shape, (100, 2), "X shape mismatch")
    assert_equal(y.shape, (100,), "y shape mismatch")
    assert_equal(np.unique(y).shape, (4,), "Unexpected number of classes")
    assert_equal(sum(y == 0), 22, "Unexpected number of samples in class #0")
    assert_equal(sum(y == 1), 31, "Unexpected number of samples in class #1")
    assert_equal(sum(y == 2), 24, "Unexpected number of samples in class #2")
    assert_equal(sum(y == 3), 23, "Unexpected number of samples in class #3")


def test_randlin():
    randlin = SG.Randlin(n_samples=100, n_features=10, n_informative=3,
            effective_rank=5, coef=True, bias=0.0, noise=1.0, random_state=0)

    X, y = randlin.regression_task()
    tasks.assert_regression(X, y, 100)
    assert_equal(X.shape, (100, 10), "X shape mismatch")
    assert_equal(y.shape, (100, 1), "y shape mismatch")

    c = randlin.ground_truth
    assert_equal(c.shape, (10,), "coef shape mismatch")
    assert_equal(sum(c != 0.0), 3, "Unexpected number of informative features")

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0)
    assert_approx_equal(np.std(y[:,0] - np.dot(X, c)), 1.0, significant=2)


def test_blobs():
    blobs = SG.Blobs(n_samples=50, n_features=2,
            centers=[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            random_state=0)
    X, y = blobs.classification_task()
    tasks.assert_classification(X, y)

    assert_equal(X.shape, (50, 2), "X shape mismatch")
    assert_equal(y.shape, (50,), "y shape mismatch")
    assert_equal(np.unique(y).shape, (3,), "Unexpected number of blobs")


def test_friedman1():
    X, y = SG.Friedman1(n_samples=5, n_features=10, noise=0.0,
                          random_state=0).regression_task()

    assert_equal(X.shape, (5, 10), "X shape mismatch")
    assert_equal(y.shape, (5, 1), "y shape mismatch")

    assert_array_almost_equal(y[:,0], 10 * np.sin(np.pi * X[:, 0] * X[:, 1])
                                 + 20 * (X[:, 2] - 0.5) ** 2 \
                                 + 10 * X[:, 3] + 5 * X[:, 4])


def test_friedman2():
    X, y = SG.Friedman2(n_samples=5, noise=0.0, random_state=0).regression_task()

    assert_equal(X.shape, (5, 4), "X shape mismatch")
    assert_equal(y.shape, (5, 1), "y shape mismatch")

    assert_array_almost_equal(y[:,0], (X[:, 0] ** 2
                                 + (X[:, 1] * X[:, 2]
                                    - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5)


def test_friedman3():
    X, y = SG.Friedman3(n_samples=5, noise=0.0, random_state=0).regression_task()

    assert_equal(X.shape, (5, 4), "X shape mismatch")
    assert_equal(y.shape, (5, 1), "y shape mismatch")

    assert_array_almost_equal(y[:,0], np.arctan((X[:, 1] * X[:, 2]
                                            - 1 / (X[:, 1] * X[:, 3]))
                                           / X[:, 0]))


def test_low_rank_matrix():
    lrm = SG.LowRankMatrix(n_samples=50, n_features=25, effective_rank=5,
                             tail_strength=0.01, random_state=0)
    X = lrm.latent_structure_task()
    tasks.assert_latent_structure(X)

    assert_equal(X.shape, (50, 25), "X shape mismatch")

    from numpy.linalg import svd
    u, s, v = svd(X)
    assert sum(s) - 5 < 0.1, "X rank is not approximately 5"

    X, Y = lrm.matrix_completion_task()
    tasks.assert_matrix_completion(X, Y)


def test_sparse_coded_signal():
    scs = SG.SparseCodedSignal(n_samples=5, n_components=8, n_features=10,
            n_nonzero_coefs=3, random_state=0)
    Y = scs.latent_structure_task()
    D = scs.D # XXX use scs.descr
    X = scs.X # XXX use scs.meta
    tasks.assert_latent_structure(Y)
    assert_equal(Y.shape, (10, 5), "Y shape mismatch")
    assert_equal(D.shape, (10, 8), "D shape mismatch")
    assert_equal(X.shape, (8, 5), "X shape mismatch")
    for col in X.T:
        assert_equal(len(np.flatnonzero(col)), 3, 'Non-zero coefs mismatch')
    assert_equal(np.dot(D, X), Y)
    assert_array_almost_equal(np.sqrt((D ** 2).sum(axis=0)),
                              np.ones(D.shape[1]))


def test_sparse_uncorrelated():
    X, y = SG.SparseUncorrelated(n_samples=5, n_features=10,
            random_state=0).regression_task()
    tasks.assert_regression(X, y)
    assert_equal(X.shape, (5, 10), "X shape mismatch")
    assert_equal(y.shape, (5, 1), "y shape mismatch")


def test_swiss_roll():
    X, t = SG.SwissRoll(n_samples=5, noise=0.0,
            random_state=0).regression_task()

    assert_equal(X.shape, (5, 3), "X shape mismatch")
    assert_equal(t.shape, (5, 1), "t shape mismatch")
    t = t[:, 0]
    assert_equal(X[:, 0], t * np.cos(t))
    assert_equal(X[:, 2], t * np.sin(t))


def test_make_s_curve():
    X, t = SG.S_Curve(n_samples=5, noise=0.0, random_state=0).regression_task()

    assert_equal(X.shape, (5, 3), "X shape mismatch")
    assert_equal(t.shape, (5, 1), "t shape mismatch")
    t = t[:, 0]
    assert_equal(X[:, 0], np.sin(t))
    assert_equal(X[:, 2], np.sign(t) * (np.cos(t) - 1))
