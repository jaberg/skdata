"""Task API
"""

def assert_classification(X, y, N=None):
    A, B = X.shape
    C, = y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'int' in str(y.dtype), y.dtype


def assert_regression(X, Y, N=None):
    A, B = X.shape
    C, D = Y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'float' in str(Y.dtype)


def assert_clustering(X, N=None):
    A, B = X.shape
    assert A == (A if N is None else N)
    assert 'float' in str(X.dtype)


def assert_factorization(X, N=None):
    A, B = X.shape
    assert A == (A if N is None else N)
    assert 'float' in str(X.dtype)


def assert_matrix_completion(X, Y, N=None):
    A, B = X.shape
    C, D = Y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'float' in str(Y.dtype)
    assert X.nnz
    assert Y.nnz

