"""Task API
"""

def assert_classification(X, y, N=None):
    A, B = X.shape
    C, = y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'int' in str(y.dtype), y.dtype


def assert_classification_train_valid_test(train, valid, test):
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test

    assert_classification(train)
    assert_classification(valid)
    assert_classification(test)

    assert X_train.shape[1] == X_valid.shape[1]
    assert X_train.shape[1] == X_test.shape[1]


def assert_regression(X, Y, N=None):
    A, B = X.shape
    C, D = Y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'float' in str(Y.dtype)


def assert_matrix_completion(X, Y, N=None):
    A, B = X.shape
    C, D = Y.shape
    assert A == C == (C if N is None else N)
    assert 'float' in str(X.dtype)
    assert 'float' in str(Y.dtype)
    assert X.nnz
    assert Y.nnz


def assert_latent_structure(X, N=None):
    A, B = X.shape
    assert A == (A if N is None else N)
    assert 'float' in str(X.dtype)


def classification_train_valid_test(dataset):
    """
    :returns: the standard train/valid/test split.
    :rtype: (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    """
    # XXX this code is note tested

    if hasattr(dataset, 'classification_train_valid_test_task'):
        return dataset.classification_train_valid_test_task()

    X, y = dataset.classification_task()

    # construct the standard splits by convention of the .meta attribute
    splits = [m['split'] for m in dataset.meta]
    train_idxs = [i for i, s in enumerate(splits) if s == 'train']
    valid_idxs = [i for i, s in enumerate(splits) if s == 'valid']
    test_idxs = [i for i, s in enumerate(splits) if s == 'test']

    if len(splits) != len(X):
        raise ValueError('Length of X does not match length of meta data')

    if len(train_idxs) + len(valid_idxs) + len(test_idxs) != len(splits):
        raise ValueError('meta contains splits other than train, valid, test.')

    X_train = larray.reindex(X, train_idxs)
    X_valid = larray.reindex(X, valid_idxs)
    X_test = larray.reindex(X, test_idxs)

    y_train = larray.reindex(y, train_idxs)
    y_valid = larray.reindex(y, valid_idxs)
    y_test = larray.reindex(y, test_idxs)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)



