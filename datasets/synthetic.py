"""
Synthetic data sets.
"""

# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Bergstra, D. Warde-Farley
# License: BSD 3 clause

# XXX: main_show would be nice to have for several of these datasets
# 
# XXX: by these datasets default to using a different random state on every call
#      - I think this is bad. Thoughts?
#
# XXX: make some of these datasets infinite to test out that lazy-evaluation
# machinery on meta data.

import numpy as np
from scipy import linalg, sparse

from .utils import check_random_state


class Base(object):
    def __init__(self, X, y=None):
        self._X = X
        self._Y = y

        if y is None:
            self.meta = [dict(x=xi) for xi in self._X]
        else:
            self.meta = [dict(x=xi, y=yi) for xi, yi in zip(self._X, self._Y)]
        self.meta_const = {}
        self.descr = {}


class Regression(object):
    def regression_task(self):
        # XXX: try this
        #      and fall back on rebuilding from self.meta
        return self._X, self._Y


class Classification(object):
    def classification_task(self):
        # XXX: try this
        #      and fall back on rebuilding from self.meta
        return self._X, self._Y


class LatentStructure(object):
    def latent_structure_task(self):
        # XXX: try this
        #      and fall back on rebuilding from self.meta
        return self._X


class Madelon(Base, Classification):
    """Random classification task.

    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    """
    def __init__(self,
            n_samples=100,
            n_features=20,
            n_informative=2,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            weights=None,
            flip_y=0.01,
            class_sep=1.0,
            hypercube=True,
            shift=0.0,
            scale=1.0,
            shuffle=True,
            random_state=None):
        """
        Generate a random n-class classification problem.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        n_features : int, optional (default=20)
            The total number of features. These comprise `n_informative`
            informative features, `n_redundant` redundant features, `n_repeated`
            dupplicated features and `n_features-n_informative-n_redundant-
            n_repeated` useless features drawn at random.

        n_informative : int, optional (default=2)
            The number of informative features. Each class is composed of a number
            of gaussian clusters each located around the vertices of a hypercube
            in a subspace of dimension `n_informative`. For each cluster,
            informative features are drawn independently from  N(0, 1) and then
            randomly linearly combined in order to add covariance. The clusters
            are then placed on the vertices of the hypercube.

        n_redundant : int, optional (default=2)
            The number of redundant features. These features are generated as
            random linear combinations of the informative features.

        n_repeated : int, optional (default=2)
            The number of dupplicated features, drawn randomly from the informative
            and the redundant features.

        n_classes : int, optional (default=2)
            The number of classes (or labels) of the classification problem.

        n_clusters_per_class : int, optional (default=2)
            The number of clusters per class.

        weights : list of floats or None (default=None)
            The proportions of samples assigned to each class. If None, then
            classes are balanced. Note that if `len(weights) == n_classes - 1`,
            then the last class weight is automatically inferred.

        flip_y : float, optional (default=0.01)
            The fraction of samples whose class are randomly exchanged.

        class_sep : float, optional (default=1.0)
            The factor multiplying the hypercube dimension.

        hypercube : boolean, optional (default=True)
            If True, the clusters are put on the vertices of a hypercube. If
            False, the clusters are put on the vertices of a random polytope.

        shift : float or None, optional (default=0.0)
            Shift all features by the specified value. If None, then features
            are shifted by a random value drawn in [-class_sep, class_sep].

        scale : float or None, optional (default=1.0)
            Multiply all features by the specified value. If None, then features
            are scaled by a random value drawn in [1, 100]. Note that scaling
            happens after shifting.

        shuffle : boolean, optional (default=True)
            Shuffle the samples and the features.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Return
        ------
        X : array of shape [n_samples, n_features]
            The generated samples.

        y : array of shape [n_samples]
            The integer labels for class membership of each sample.
        """
        generator = check_random_state(random_state)

        # Count features, clusters and samples
        assert n_informative + n_redundant + n_repeated <= n_features
        assert 2 ** n_informative >= n_classes * n_clusters_per_class
        assert weights is None or (len(weights) == n_classes or
                                   len(weights) == (n_classes - 1))

        n_useless = n_features - n_informative - n_redundant - n_repeated
        n_clusters = n_classes * n_clusters_per_class

        if weights and len(weights) == (n_classes - 1):
            weights.append(1.0 - sum(weights))

        if weights is None:
            weights = [1.0 / n_classes] * n_classes
            weights[-1] = 1.0 - sum(weights[:-1])

        n_samples_per_cluster = []

        for k in xrange(n_clusters):
            n_samples_per_cluster.append(int(n_samples * weights[k % n_classes]
                                         / n_clusters_per_class))

        for i in xrange(n_samples - sum(n_samples_per_cluster)):
            n_samples_per_cluster[i % n_clusters] += 1

        # Intialize X and y
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype='int')

        # Build the polytope
        from itertools import product
        C = np.array(list(product([-class_sep, class_sep], repeat=n_informative)))

        if not hypercube:
            for k in xrange(n_clusters):
                C[k, :] *= generator.rand()

            for f in xrange(n_informative):
                C[:, f] *= generator.rand()

        generator.shuffle(C)

        # Loop over all clusters
        pos = 0
        pos_end = 0

        for k in xrange(n_clusters):
            # Number of samples in cluster k
            n_samples_k = n_samples_per_cluster[k]

            # Define the range of samples
            pos = pos_end
            pos_end = pos + n_samples_k

            # Assign labels
            y[pos:pos_end] = k % n_classes

            # Draw features at random
            X[pos:pos_end, :n_informative] = generator.randn(n_samples_k,
                                                             n_informative)

            # Multiply by a random matrix to create co-variance of the features
            A = 2 * generator.rand(n_informative, n_informative) - 1
            X[pos:pos_end, :n_informative] = np.dot(X[pos:pos_end, :n_informative],
                                                    A)

            # Shift the cluster to a vertice
            X[pos:pos_end, :n_informative] += np.tile(C[k, :], (n_samples_k, 1))

        # Create redundant features
        if n_redundant > 0:
            B = 2 * generator.rand(n_informative, n_redundant) - 1
            X[:, n_informative:n_informative + n_redundant] = \
                                                np.dot(X[:, :n_informative], B)

        # Repeat some features
        if n_repeated > 0:
            n = n_informative + n_redundant
            indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.int)
            X[:, n:n + n_repeated] = X[:, indices]

        # Fill useless features
        X[:, n_features - n_useless:] = generator.randn(n_samples, n_useless)

        # Randomly flip labels
        if flip_y >= 0.0:
            for i in xrange(n_samples):
                if generator.rand() < flip_y:
                    y[i] = generator.randint(n_classes)

        # Randomly shift and scale
        constant_shift = shift is not None
        constant_scale = scale is not None

        for f in xrange(n_features):
            if not constant_shift:
                shift = (2 * generator.rand() - 1) * class_sep

            if not constant_scale:
                scale = 1 + 100 * generator.rand()

            X[:, f] += shift
            X[:, f] *= scale

        # Randomly permute samples and features
        if shuffle:
            indices = range(n_samples)
            generator.shuffle(indices)
            X = X[indices]
            y = y[indices]

            indices = range(n_features)
            generator.shuffle(indices)
            X[:, :] = X[:, indices]

        Base.__init__(self, X, y)


class FourRegions(Base, Classification):
    """The four regions classification task.

    A classic benchmark task for non-linear classifiers. Generates
    a 2-dimensional dataset on the [-1, 1]^2 square where two
    concentric rings are divided in half, and opposing sides of
    the inner and outer circles are assigned to the same class,
    with two more classes formed from the two halves of the square
    excluding the rings.

    References
    ----------

    .. [1] S. Singhal and L. Wu, "Training multilayer perceptrons
           with the extended Kalman algorithm". Advances in Neural
           Information Processing Systems, Proceedings of the 1988
           Conference, pp.133-140.
           http://books.nips.cc/papers/files/nips01/0133.pdf
    """
    def __init__(self, n_samples=100, n_features=2, random_state=None):
        """Generate a (finite) dataset for the four regions task.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate in this instance of the
            dataset.

        n_features : int, optional
            The number of features (dimensionality of the task). The
            default, 2, recovers the standard four regions task, but
            the task can be meaningfully generalized to higher
            dimensions (though the class balance will change).

        random_state : int, RandomState instance or None, optional
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the
            random number generator; If None, the random number
            generator is the RandomState instance used by `np.random`.
        """
        assert n_features >= 2, ("Cannot generate FourRegions dataset with "
                                 "n_features < 2")
        generator = check_random_state(random_state)
        X = generator.uniform(-1, 1, size=(n_samples, n_features))
        y = -np.ones(X.shape[0], dtype=int)
        top_half = X[:, 1] > 0
        right_half = X[:, 0] > 0
        dists = np.sqrt(np.sum(X ** 2, axis=1))

        # The easy ones -- the outer shelf.
        outer = dists > 5. / 6.
        y[np.logical_and(top_half, outer)] = 2
        y[np.logical_and(np.logical_not(top_half), outer)] = 3
        first_ring = np.logical_and(dists > 1. / 6., dists <= 1. / 2.)
        second_ring = np.logical_and(dists > 1. / 2., dists <= 5. / 6.)

        # Region 2 -- right inner and left outer, excluding center nut
        y[np.logical_and(first_ring, right_half)] = 1
        y[np.logical_and(second_ring, np.logical_not(right_half))] = 1

        # Region 1 -- left inner and right outer, including center nut
        y[np.logical_and(second_ring, right_half)] = 0
        y[np.logical_and(np.logical_not(right_half), dists < 1. / 2.)] = 0
        y[np.logical_and(right_half, dists < 1. / 6.)] = 0

        assert np.all(y >= 0)
        Base.__init__(self, X, y)

    @classmethod
    def main_show(cls, n_samples=50000):
        dataset = cls(n_samples=n_samples)
        import matplotlib.pyplot as plt
        X, y = dataset.classification_task()
        plt.scatter(X[:, 0], X[:, 1], 10, y, cmap='gray')
        plt.axis('equal')
        plt.title('%d samples from the four regions task' % len(X))
        plt.show()


class Randlin(Base, Regression):
    """Random linear regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See the `make_low_rank_matrix` for
    more details.

    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.
    """
    def __init__(self, n_samples=100, n_features=100, n_informative=10,
            bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0,
            shuffle=True, coef=False, random_state=None):
        """

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        n_features : int, optional (default=100)
            The number of features.

        n_informative : int, optional (default=10)
            The number of informative features, i.e., the number of features used
            to build the linear model used to generate the output.

        bias : float, optional (default=0.0)
            The bias term in the underlying linear model.

        effective_rank : int or None, optional (default=None)
            if not None:
                The approximate number of singular vectors required to explain most
                of the input data by linear combinations. Using this kind of
                singular spectrum in the input allows the generator to reproduce
                the correlations often observed in practice.
            if None:
                The input set is well conditioned, centered and gaussian with
                unit variance.

        tail_strength : float between 0.0 and 1.0, optional (default=0.5)
            The relative importance of the fat noisy tail of the singular values
            profile if `effective_rank` is not None.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise applied to the output.

        shuffle : boolean, optional (default=True)
            Shuffle the samples and the features.

        coef : boolean, optional (default=False)
            If True, the coefficients of the underlying linear model are returned.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The output values.

        coef : array of shape [n_features], optional
            The coefficient of the underlying linear model. It is returned only if
            coef is True.
        """
        generator = check_random_state(random_state)

        if effective_rank is None:
            # Randomly generate a well conditioned input set
            X = generator.randn(n_samples, n_features)

        else:
            # Randomly generate a low rank, fat tail input set
            X = LowRankMatrix(n_samples=n_samples,
                                     n_features=n_features,
                                     effective_rank=effective_rank,
                                     tail_strength=tail_strength,
                                     random_state=generator)._X

        # Generate a ground truth model with only n_informative features being non
        # zeros (the other features are not correlated to y and should be ignored
        # by a sparsifying regularizers such as L1 or elastic net)
        ground_truth = np.zeros(n_features)
        ground_truth[:n_informative] = 100 * generator.rand(n_informative)

        y = np.dot(X, ground_truth) + bias

        # Add noise
        if noise > 0.0:
            y += generator.normal(scale=noise, size=y.shape)

        # Randomly permute samples and features
        if shuffle:
            indices = range(n_samples)
            generator.shuffle(indices)
            X = X[indices]
            y = y[indices]

            indices = range(n_features)
            generator.shuffle(indices)
            X[:, :] = X[:, indices]
            ground_truth = ground_truth[indices]

        Base.__init__(self, X, y[:, None])
        self.ground_truth = ground_truth


class Blobs(Base, Classification, LatentStructure):
    """Generate isotropic Gaussian blobs for clustering.
    """
    def __init__(self, n_samples=100, n_features=2, centers=3, cluster_std=1.0,
                   center_box=(-10.0, 10.0), shuffle=True, random_state=None):
        """

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points equally divided among clusters.

        n_features : int, optional (default=2)
            The number of features for each sample.

        centers : int or array of shape [n_centers, n_features], optional (default=3)
            The number of centers to generate, or the fixed center locations.

        cluster_std: float or sequence of floats, optional (default=1.0)
            The standard deviation of the clusters.

        center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))
            The bounding box for each cluster center when centers are
            generated at random.

        shuffle : boolean, optional (default=True)
            Shuffle the samples.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Return
        ------
        X : array of shape [n_samples, n_features]
            The generated samples.

        y : array of shape [n_samples]
            The integer labels for cluster membership of each sample.

        """
        generator = check_random_state(random_state)

        if isinstance(centers, int):
            centers = generator.uniform(center_box[0], center_box[1],
                                        size=(centers, n_features))
        else:
            centers = np.atleast_2d(centers)
            n_features = centers.shape[1]

        X = []
        y = []

        n_centers = centers.shape[0]
        n_samples_per_center = [n_samples / n_centers] * n_centers

        for i in xrange(n_samples % n_centers):
            n_samples_per_center[i] += 1

        for i, n in enumerate(n_samples_per_center):
            X.append(centers[i] + generator.normal(scale=cluster_std,
                                                   size=(n, n_features)))
            y += [i] * n

        X = np.concatenate(X)
        y = np.array(y)

        if shuffle:
            indices = np.arange(n_samples)
            generator.shuffle(indices)
            X = X[indices]
            y = y[indices]

        Base.__init__(self, X, y)

    @classmethod
    def main_show(cls):
        self = cls(n_samples=500)
        import matplotlib.pyplot as plt
        plt.scatter(self._X[:, 0], self._X[:, 1])
        plt.show()


class Friedman1(Base, Regression):
    def __init__(self, n_samples=100, n_features=10, noise=0.0, random_state=None):
        """
        Generate the "Friedman #1" regression problem as described in Friedman [1]
        and Breiman [2].

        Inputs `X` are independent features uniformly distributed on the interval
        [0, 1]. The output `y` is created according to the formula::

            y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
                   + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

        Out of the `n_features` features, only 5 are actually used to compute
        `y`. The remaining features are independent of `y`.

        The number of features has to be >= 5.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        n_features : int, optional (default=10)
            The number of features. Should be at least 5.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise applied to the output.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The output values.

        References
        ----------
        .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
               of Statistics 19 (1), pages 1-67, 1991.

        .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
               pages 123-140, 1996.
        """
        assert n_features >= 5

        generator = check_random_state(random_state)

        X = generator.rand(n_samples, n_features)
        y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
            + 10 * X[:, 3] + 5 * X[:, 4] + noise * generator.randn(n_samples)

        Base.__init__(self, X, y[:, None])


class Friedman2(Base, Regression):
    def __init__(self, n_samples=100, noise=0.0, random_state=None):
        """
        Generate the "Friedman #2" regression problem as described in Friedman [1]
        and Breiman [2].

        Inputs `X` are 4 independent features uniformly distributed on the
        intervals::

            0 <= X[:, 0] <= 100,
            40 * pi <= X[:, 1] <= 560 * pi,
            0 <= X[:, 2] <= 1,
            1 <= X[:, 3] <= 11.

        The output `y` is created according to the formula::

            y(X) = (X[:, 0] ** 2 \
                       + (X[:, 1] * X[:, 2] \
                             - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 \
                   + noise * N(0, 1).

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise applied to the output.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, 4]
            The input samples.

        y : array of shape [n_samples]
            The output values.

        References
        ----------
        .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
               of Statistics 19 (1), pages 1-67, 1991.

        .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
               pages 123-140, 1996.
        """
        generator = check_random_state(random_state)

        X = generator.rand(n_samples, 4)
        X[:, 0] *= 100
        X[:, 1] *= 520 * np.pi
        X[:, 1] += 40 * np.pi
        X[:, 3] *= 10
        X[:, 3] += 1

        y = (X[:, 0] ** 2
                + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 \
            + noise * generator.randn(n_samples)

        return Base.__init__(self, X, y[:, None])


class Friedman3(Base, Regression):
    def __init__(self, n_samples=100, noise=0.0, random_state=None):
        """
        Generate the "Friedman #3" regression problem as described in Friedman [1]
        and Breiman [2].

        Inputs `X` are 4 independent features uniformly distributed on the
        intervals::

            0 <= X[:, 0] <= 100,
            40 * pi <= X[:, 1] <= 560 * pi,
            0 <= X[:, 2] <= 1,
            1 <= X[:, 3] <= 11.

        The output `y` is created according to the formula::

            y(X) = arctan((X[:, 1] * X[:, 2] \
                              - 1 / (X[:, 1] * X[:, 3])) \
                          / X[:, 0]) \
                   + noise * N(0, 1).

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise applied to the output.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, 4]
            The input samples.

        y : array of shape [n_samples]
            The output values.

        References
        ----------
        .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
               of Statistics 19 (1), pages 1-67, 1991.

        .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
               pages 123-140, 1996.
        """
        generator = check_random_state(random_state)

        X = generator.rand(n_samples, 4)
        X[:, 0] *= 100
        X[:, 1] *= 520 * np.pi
        X[:, 1] += 40 * np.pi
        X[:, 3] *= 10
        X[:, 3] += 1

        y = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) \
            + noise * generator.randn(n_samples)

        Base.__init__(self, X, y[:, None])


class LowRankMatrix(Base, LatentStructure):
    """Mostly low rank random matrix with bell-shaped singular values profile.

    Most of the variance can be explained by a bell-shaped curve of width
    effective_rank: the low rank part of the singular values profile is::

        (1 - tail_strength) * exp(-1.0 * (i / effective_rank) ** 2)

    The remaining singular values' tail is fat, decreasing as::

        tail_strength * exp(-0.1 * i / effective_rank).

    The low rank part of the profile can be considered the structured
    signal part of the data while the tail can be considered the noisy
    part of the data that cannot be summarized by a low number of linear
    components (singular vectors).

    This kind of singular profiles is often seen in practice, for instance:
     - graw level pictures of faces
     - TF-IDF vectors of text documents crawled from the web
    """
    def __init__(self, n_samples=100, n_features=100, effective_rank=10,
                             tail_strength=0.5, random_state=None):
        """

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        n_features : int, optional (default=100)
            The number of features.

        effective_rank : int, optional (default=10)
            The approximate number of singular vectors required to explain most of
            the data by linear combinations.

        tail_strength : float between 0.0 and 1.0, optional (default=0.5)
            The relative importance of the fat noisy tail of the singular values
            profile.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The matrix.
        """
        generator = check_random_state(random_state)
        n = min(n_samples, n_features)

        # Random (ortho normal) vectors
        from .utils import qr_economic
        u, _ = qr_economic(generator.randn(n_samples, n))
        v, _ = qr_economic(generator.randn(n_features, n))

        # Index of the singular values
        singular_ind = np.arange(n, dtype=np.float64)

        # Build the singular profile by assembling signal and noise components
        low_rank = (1 - tail_strength) * \
                   np.exp(-1.0 * (singular_ind / effective_rank) ** 2)
        tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
        s = np.identity(n) * (low_rank + tail)

        Base.__init__(self, np.dot(np.dot(u, s), v.T))

        self.descr['mask'] = generator.randint(3, size=self._X.shape)

    def matrix_completion_task(self):
        X = sparse.csr_matrix(self._X * (self.descr['mask'] == 0))
        Y = sparse.csr_matrix(self._X * (self.descr['mask'] == 1))
        assert X.nnz == (self.descr['mask'] == 0).sum()
        assert Y.nnz == (self.descr['mask'] == 1).sum()
        # where mask is 2 is neither in X nor Y
        return X, Y


class SparseCodedSignal(Base, LatentStructure):
    """Generate a signal as a sparse combination of dictionary elements.

    Returns a matrix Y = DX, such as D is (n_features, n_components),
    X is (n_components, n_samples) and each column of X has exactly
    n_nonzero_coefs non-zero elements.

    """
    def __init__(self, n_samples, n_components, n_features, n_nonzero_coefs,
            random_state=None):
        """
        Parameters
        ----------
        n_samples : int
            number of samples to generate

        n_components:  int,
            number of components in the dictionary

        n_features : int
            number of features of the dataset to generate

        n_nonzero_coefs : int
            number of active (non-zero) coefficients in each sample

        random_state: int or RandomState instance, optional (default=None)
            seed used by the pseudo random number generator

        Returns
        -------
        data: array of shape [n_features, n_samples]
            The encoded signal (Y).

        dictionary: array of shape [n_features, n_components]
            The dictionary with normalized components (D).

        code: array of shape [n_components, n_samples]
            The sparse code such that each column of this matrix has exactly
            n_nonzero_coefs non-zero items (X).

        """
        generator = check_random_state(random_state)

        # generate dictionary
        D = generator.randn(n_features, n_components)
        D /= np.sqrt(np.sum((D ** 2), axis=0))

        # generate code
        X = np.zeros((n_components, n_samples))
        for i in xrange(n_samples):
            idx = np.arange(n_components)
            generator.shuffle(idx)
            idx = idx[:n_nonzero_coefs]
            X[idx, i] = generator.randn(n_nonzero_coefs)

        # XXX: self.meta should include list of non-zeros in X
        # XXX: self.descr should include dictionary D
        self.D = D
        self.X = X
        Base.__init__(self, np.dot(D, X))


class SparseUncorrelated(Base, Regression):
    """Generate a random regression problem with sparse uncorrelated design
    as described in Celeux et al [1].::

        X ~ N(0, 1)
        y(X) = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]

    Only the first 4 features are informative. The remaining features are
    useless.

    References
    ----------
    .. [1] G. Celeux, M. El Anbari, J.-M. Marin, C. P. Robert,
           "Regularization in regression: comparing Bayesian and frequentist
           methods in a poorly informative situation", 2009.

    """
    def __init__(self, n_samples=100, n_features=10, random_state=None):
        """
        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of samples.

        n_features : int, optional (default=10)
            The number of features.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        """
        generator = check_random_state(random_state)
        X = generator.normal(loc=0, scale=1, size=(n_samples, n_features))
        y = generator.normal(loc=(X[:, 0] +
                                  2 * X[:, 1] -
                                  2 * X[:, 2] -
                                  1.5 * X[:, 3]), scale=np.ones(n_samples))
        Base.__init__(self, X, y[:, None])


class SwissRoll(Base, Regression, LatentStructure):
    """Generate a swiss roll dataset.

    Notes
    -----
    The algorithm is from Marsland [1].

    References
    ----------
    .. [1] S. Marsland, "Machine Learning: An Algorithmic Perpsective",
           Chapter 10, 2009.
           http://www-ist.massey.ac.nz/smarsland/Code/10/lle.py
    """
    def __init__(self, n_samples=100, noise=0.0, random_state=None):
        """
        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of sample points on the S curve.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        X : array of shape [n_samples, 3]
            The points.

        t : array of shape [n_samples]
            The univariate position of the sample according to the main dimension
            of the points in the manifold.
        """
        generator = check_random_state(random_state)

        t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
        x = t * np.cos(t)
        y = 21 * generator.rand(1, n_samples)
        z = t * np.sin(t)

        X = np.concatenate((x, y, z))
        X += noise * generator.randn(3, n_samples)
        X = X.T
        t = np.squeeze(t)
        Base.__init__(self, X, t[:, None])

    @classmethod
    def main_show(cls):
        self = cls(n_samples=1000)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self._X[:, 2], self._X[:, 1], self._X[:, 0])
        plt.show()


class S_Curve(Base, Regression, LatentStructure):
    """Generate an S curve dataset.
    """
    def __init__(self, n_samples=100, noise=0.0, random_state=None):
        """

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The number of sample points on the S curve.

        noise : float, optional (default=0.0)
            The standard deviation of the gaussian noise.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        """
        generator = check_random_state(random_state)

        t = 3 * np.pi * (generator.rand(1, n_samples) - 0.5)
        x = np.sin(t)
        y = 2.0 * generator.rand(1, n_samples)
        z = np.sign(t) * (np.cos(t) - 1)

        X = np.concatenate((x, y, z))
        X += noise * generator.randn(3, n_samples)
        X = X.T
        t = np.squeeze(t)

        Base.__init__(self, X, t[:, None])

    @classmethod
    def main_show(cls):
        self = cls(n_samples=1000)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self._X[:, 2], self._X[:, 1], self._X[:, 0])
        plt.show()


