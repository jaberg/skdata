from .dotdict import *
from .image import *

from .my_path import get_my_path, get_my_path_basename
from .xml2x import xml2dict, xml2list
from .download_and_extract import download, extract, download_and_extract


# -- old utils.py
import numpy as np
import scipy.sparse as sp
import warnings
import scipy.linalg

_FLOAT_CODES = np.typecodes['AllFloat']


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.
    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    # O(n) time, O(1) solution. XXX: will fail if the sum over X is
    # *extremely* large. A proper solution would be a C-level loop to check
    # each element.
    if X.dtype.char in _FLOAT_CODES and not np.isfinite(X.sum()):
        raise ValueError("array contains NaN or infinity")


def safe_asanyarray(X, dtype=None, order=None):
    if not sp.issparse(X):
        X = np.asanyarray(X, dtype, order)
    assert_all_finite(X)
    return X


def as_float_array(X, overwrite_X=False):
    """
    Converts a numpy array to type np.float

    The new dtype will be float32 or np.float64, depending on the original
    type.  The function can create a copy or modify the argument depending of
    the argument overwrite_X

    WARNING : If X is not of type float, then a copy of X with the right type
              will be returned

    Parameters
    ----------
    X : array

    overwrite_X : bool, optional
        if False, a copy of X will be created

    Returns
    -------
    X : array
        An array of type np.float
    """
    if X.dtype in [np.float32, np.float64]:
        if overwrite_X:
            return X
        else:
            return X.copy()
    if X.dtype == np.int32:
        X = X.astype(np.float32)
    else:
        X = X.astype(np.float64)
    return X


def atleast2d_or_csr(X):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format"""
    X = X.tocsr() if sp.issparse(X) else np.atleast_2d(X)
    assert_all_finite(X)
    return X


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_arrays(*arrays, **options):
    """Checked that all arrays have consistent first dimensions

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays.

    sparse_format : 'csr' or 'csc', None by default
        If not None, any scipy.sparse matrix is converted to
        Compressed Sparse Rows or Compressed Sparse Columns representations.

    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).
    """
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    if options:
        raise ValueError("Unexpected kw arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    first = arrays[0]
    if not hasattr(first, '__len__') and not hasattr(first, 'shape'):
        raise ValueError("Expected python sequence or array, got %r" % first)
    n_samples = first.shape[0] if hasattr(first, 'shape') else len(first)

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue

        if not hasattr(array, '__len__') and not hasattr(array, 'shape'):
            raise ValueError("Expected python sequence or array, got %r"
                             % array)
        size = array.shape[0] if hasattr(array, 'shape') else len(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d" % (
                size, n_samples))

        if sp.issparse(array):
            if sparse_format == 'csr':
                array = array.tocsr()
            elif sparse_format == 'csc':
                array = array.tocsc()
        else:
            array = np.asanyarray(array)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays


def warn_if_not_float(X, estimator='This algorithm'):
    """Warning utility function to check that data type is floating point"""
    if not isinstance(estimator, basestring):
        estimator = estimator.__class__.__name__
    if X.dtype.kind != 'f':
        warnings.warn("%s assumes floating point values as input, "
                      "got %s" % (estimator, X.dtype))


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Prints a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from scikits.learn.utils import deprecated
    >>> @deprecated()
    ... def some_function(): pass

    Deprecating a class takes some work, since we want to run on Python
    versions that do not have class decorators:

    >>> class Foo(object): pass
    ...
    >>> Foo = deprecated("Use Bar instead")(Foo)
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):
        self.extra = extra

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        what = "Function %s" % fun.__name__

        msg = "%s is deprecated" % what
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


def resample(*arrays, **options):
    """Resample arrays or sparse matrices in a consistent way

    The default strategy implements one step of the bootstrapping
    procedure.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]

    replace : boolean, True by default
        Implements resampling with replacement. If False, this will implement
        (sliced) random permutations.

    n_samples : int, None by default
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.

    random_state : int or RandomState instance
        Control the shuffling for reproducible behavior.

    Return
    ------
    Sequence of resampled views of the collections. The original arrays are
    not impacted.

    Example
    -------
    It is possible to mix sparse and dense arrays in the same run::

      >>> X = [[1., 0.], [2., 1.], [0., 0.]]
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_matrix
      >>> X_sparse = coo_matrix(X)

      >>> from scikits.learn.utils import resample
      >>> X, X_sparse, y = resample(X, X_sparse, y, random_state=0)
      >>> X
      array([[ 1.,  0.],
             [ 2.,  1.],
             [ 1.,  0.]])

      >>> X_sparse                            # doctest: +NORMALIZE_WHITESPACE
      <3x2 sparse matrix of type '<type 'numpy.float64'>'
          with 4 stored elements in Compressed Sparse Row format>

      >>> X_sparse.toarray()
      array([[ 1.,  0.],
             [ 2.,  1.],
             [ 1.,  0.]])

      >>> y
      array([0, 1, 0])

      >>> resample(y, n_samples=2, random_state=0)
      array([0, 1])


    See also
    --------
    :class:`scikits.learn.cross_val.Bootstrap`
    :func:`scikits.learn.utils.shuffle`
    """
    random_state = check_random_state(options.pop('random_state', None))
    replace = options.pop('replace', True)
    max_n_samples = options.pop('n_samples', None)
    if options:
        raise ValueError("Unexpected kw arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, 'shape') else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples

    if max_n_samples > n_samples:
        raise ValueError("Cannot sample %d out of arrays with dim %d" % (
            max_n_samples, n_samples))

    # To cope with Python 2.5 syntax limitations
    kwargs = dict(sparse_format='csr')
    arrays = check_arrays(*arrays, **kwargs)

    if replace:
        indices = random_state.randint(0, n_samples, size=(max_n_samples,))
    else:
        indices = np.arange(n_samples)
        random_state.shuffle(indices)
        indices = indices[:max_n_samples]

    resampled_arrays = []

    for array in arrays:
        array = array[indices]
        resampled_arrays.append(array)

    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays


def shuffle(*arrays, **options):
    """Shuffle arrays or sparse matrices in a consistent way

    This is a convenience alias to resample(*arrays, replace=False) to do
    random permutations of the collections.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]

    random_state : int or RandomState instance
        Control the shuffling for reproducible behavior.

    n_samples : int, None by default
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.

    Return
    ------
    Sequence of shuffled views of the collections. The original arrays are
    not impacted.

    Example
    -------
    It is possible to mix sparse and dense arrays in the same run::

      >>> X = [[1., 0.], [2., 1.], [0., 0.]]
      >>> y = np.array([0, 1, 2])

      >>> from scipy.sparse import coo_matrix
      >>> X_sparse = coo_matrix(X)

      >>> from scikits.learn.utils import shuffle
      >>> X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
      >>> X
      array([[ 0.,  0.],
             [ 2.,  1.],
             [ 1.,  0.]])

      >>> X_sparse                            # doctest: +NORMALIZE_WHITESPACE
      <3x2 sparse matrix of type '<type 'numpy.float64'>'
          with 3 stored elements in Compressed Sparse Row format>

      >>> X_sparse.toarray()
      array([[ 0.,  0.],
             [ 2.,  1.],
             [ 1.,  0.]])

      >>> y
      array([2, 1, 0])

      >>> shuffle(y, n_samples=2, random_state=0)
      array([0, 1])

    See also
    --------
    :func:`scikits.learn.utils.resample`
    """
    options['replace'] = False
    return resample(*arrays, **options)


def gen_even_slices(n, n_packs):
    """Generator to create n_packs slices going up to n.

    Examples
    --------
    >>> from scikits.learn.utils import gen_even_slices
    >>> list(gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(gen_even_slices(10, 10))                     #doctest: +ELLIPSIS
    [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
    >>> list(gen_even_slices(10, 5))                      #doctest: +ELLIPSIS
    [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
    >>> list(gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield slice(start, end, None)
            start = end


def qr_economic(A, **kwargs):
    """Compat function for the QR-decomposition in economic mode

    Scipy 0.9 changed the keyword econ=True to mode='economic'
    """
    # trick: triangular solve has introduced in 0.9
    if hasattr(scipy.linalg, 'solve_triangular'):
        return scipy.linalg.qr(A, mode='economic', **kwargs)
    else:
        return scipy.linalg.qr(A, econ=True, **kwargs)


def memoize(f):
    """Simple decorator hashes *args to f, stores rvals in dict.

    This simple decorator works in memory only, does not persist between
    processes.
    """
    cache = {}
    def cache_f(*args):
        if args in cache:
            return cache[args]
        rval = f(*args)
        cache[args] = rval
        return rval
    return cache_f


def int_labels(labels, return_dct=False):
    """['me', 'b', 'b', ...] -> [0, 1, 1, ...]"""
    u = np.unique(labels)
    i = np.searchsorted(u, labels)
    if return_dct:
        return i, u
    else:
        return i


def random_spd_matrix(n_dim, random_state=None):
    """
    Generate a random symmetric, positive-definite matrix.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_dim, n_dim]
        The random symmetric, positive-definite matrix.
    """
    generator = check_random_state(random_state)

    A = generator.rand(n_dim, n_dim)
    U, s, V = scipy.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.rand(n_dim))), V)

    return X


def random_patches(images, N, rows, cols, rng, channel_major=False):
    """Return a stack of N image patches

    Parameters
    ----------
    images: array of shape (n_images, n_channels, img_rows, img_cols)
            if channel_major else
            array of shape (n_images, img_rows, img_cols, n_channels)
    N: int
        number of patches to draw

    rows: int
        pixel rows per patch

    cols: int
        pixel cols per patch

    rng: numpy.RandomState
        patch source locations are drawn uniformly

    channel_major: bool
        interpretation of `images` shape


    Returns
    -------
        array of shape (N, n_channels, rows, cols)
        if channel_major else
        array of shape (N, rows, cols, n_channels)

        A tensor of patches drawn uniformly and at random from the `images`
    
    """

    if channel_major:
        n_imgs, iF, iR, iC = images.shape
        rval = np.empty((N, iF, rows, cols), dtype=images.dtype)
    else:
        n_imgs, iR, iC, iF = images.shape
        rval = np.empty((N, rows, cols, iF), dtype=images.dtype)

    srcs = rng.randint(n_imgs, size=N)

    if rows > iR or cols > iC:
        raise ValueError('cannot extract patches', (R, C))

    roffsets = rng.randint(iR - rows + 1, size=N)
    coffsets = rng.randint(iC - cols + 1, size=N)
    # TODO: this can be done with one advanced index right?
    for rv_i, src_i, ro, co in zip(rval, srcs, roffsets, coffsets):
        if channel_major:
            rv_i[:] = images[src_i, :, ro: ro + rows, co : co + cols]
        else:
            rv_i[:] = images[src_i, ro: ro + rows, co : co + cols]
    return rval
