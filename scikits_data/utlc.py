"""
user should use the load _ndarray_dataset or load_sparse_dataset function

See the file ${PYLEARN_DATA_ROOT}/UTCL/README for detail on the datasets.

See the end of this file for an example.
"""

import cPickle
import gzip
import os

import numpy
import theano

import pylearn.io.filetensor as ft
import config

def load_ndarray_dataset(name, normalize=True, transfer=False,
                         normalize_on_the_fly=False, randomize_valid=False,
                         randomize_test=False):
    """ Load the train,valid,test data for the dataset `name`
        and return it in ndarray format.

        We suppose the data was created with ift6266h11/pretraitement/to_npy.py that
        shuffle the train. So the train should already be shuffled.

    :param normalize: If True, we normalize the train dataset
                      before returning it
    :param transfer: If True also return the transfer labels
    :param normalize_on_the_fly: If True, we return a Theano Variable that will give
                                 as output the normalized value. If the user only
                                 take a subtensor of that variable, Theano optimization
                                 should make that we will only have in memory the subtensor
                                 portion that is computed in normalized form. We store
                                 the original data in shared memory in its original dtype.

                                 This is usefull to have the original data in its original
                                 dtype in memory to same memory. Especialy usefull to
                                 be able to use rita and harry with 1G per jobs.
    :param randomize_valid: Do we randomize the order of the valid set?
                            We always use the same random order
                            If False, return in the same order as downloaded on the web
    :param randomize_test: Do we randomize the order of the test set?
                           We always use the same random order
                           If False, return in the same order as downloaded on the web
    """
    assert not (normalize and normalize_on_the_fly), "Can't normalize in 2 way at the same time!"

    assert name in ['avicenna','harry','rita','sylvester','ule']
    common = os.path.join('UTLC','filetensor',name+'_')
    trname,vname,tename = [config.get_filepath_in_roots(common+subset+'.ft.gz',
                                                        common+subset+'.ft')
                           for subset in ['train','valid','test']]

    train = load_filetensor(trname)
    valid = load_filetensor(vname)
    test = load_filetensor(tename)
    if randomize_valid:
        rng = numpy.random.RandomState([1,2,3,4])
        perm = rng.permutation(valid.shape[0])
        valid = valid[perm]
    if randomize_test:
        rng = numpy.random.RandomState([1,2,3,4])
        perm = rng.permutation(test.shape[0])
        test = test[perm]

    if normalize or normalize_on_the_fly:
        if normalize_on_the_fly:
            # Shared variables of the original type
            train = theano.shared(train, borrow=True, name=name+"_train")
            valid = theano.shared(valid, borrow=True, name=name+"_valid")
            test = theano.shared(test, borrow=True, name=name+"_test")
            # Symbolic variables cast into floatX
            train = theano.tensor.cast(train, theano.config.floatX)
            valid = theano.tensor.cast(valid, theano.config.floatX)
            test = theano.tensor.cast(test, theano.config.floatX)
        else:
            train = numpy.asarray(train, theano.config.floatX)
            valid = numpy.asarray(valid, theano.config.floatX)
            test = numpy.asarray(test, theano.config.floatX)

        if name == "ule":
            train /= 255
            valid /= 255
            test /= 255
        elif name in ["avicenna", "sylvester"]:
            if name == "avicenna":
                train_mean = 514.62154022835455
                train_std = 6.829096494224145
            else:
                train_mean = 403.81889927027686
                train_std = 96.43841050784053
            train -= train_mean
            valid -= train_mean
            test -= train_mean
            train /= train_std
            valid /= train_std
            test /= train_std
        elif name == "harry":
            std = 0.69336046033925791#train.std()slow to compute
            train /= std
            valid /= std
            test /= std
        elif name == "rita":
            v = numpy.asarray(230, dtype=theano.config.floatX)
            train /= v
            valid /= v
            test /= v
        else:
            raise Exception("This dataset don't have its normalization defined")
    if transfer:
        transfer = load_filetensor(os.path.join(config.data_root(),"UTLC","filetensor",name+"_transfer.ft"))
        return train, valid, test, transfer
    else:
        return train, valid, test

def load_sparse_dataset(name, normalize=True, transfer=False,
                        randomize_valid=False,
                        randomize_test=False):
    """ Load the train,valid,test data for the dataset `name`
        and return it in sparse format.

        We suppose the data was created with ift6266h11/pretraitement/to_npy.py that
        shuffle the train. So the train should already be shuffled.

    :param normalize: If True, we normalize the train dataset
                      before returning it
    :param transfer: If True also return the transfer label
    :param randomize_valid: see same option for load_ndarray_dataset
    :param randomize_test: see same option for load_ndarray_dataset

    """
    assert name in ['harry','terry','ule']
    common = os.path.join('UTLC','sparse',name+'_')
    trname,vname,tename = [config.get_filepath_in_roots(common+subset+'.npy.gz',
                                                        common+subset+'.npy')
                           for subset in ['train','valid','test']]
    train = load_sparse(trname)
    valid = load_sparse(vname)
    test = load_sparse(tename)

    # Data should already be in csr format that support
    # this type of indexing.
    if randomize_valid:
        rng = numpy.random.RandomState([1,2,3,4])
        perm = rng.permutation(valid.shape[0])
        valid = valid[perm]
    if randomize_test:
        rng = numpy.random.RandomState([1,2,3,4])
        perm = rng.permutation(test.shape[0])
        test = test[perm]

    if normalize:
        if name == "ule":
            train = train.astype(theano.config.floatX) / 255
            valid = valid.astype(theano.config.floatX) / 255
            test = test.astype(theano.config.floatX) / 255
        elif name == "harry":
            train = train.astype(theano.config.floatX)
            valid = valid.astype(theano.config.floatX)
            test = test.astype(theano.config.floatX)
            std = 0.69336046033925791#train.std()slow to compute
            train = (train) / std
            valid = (valid) / std
            test = (test) / std
        elif name == "terry":
            train = train.astype(theano.config.floatX)
            valid = valid.astype(theano.config.floatX)
            test = test.astype(theano.config.floatX)
            train = (train) / 300
            valid = (valid) / 300
            test = (test) / 300
        else:
            raise Exception("This dataset don't have its normalization defined")
    if transfer:
        transfer = load_filetensor(os.path.join(config.data_root(),"UTLC","filetensor",name+"_transfer.ft"))
        return train, valid, test, transfer
    else:
        return train, valid, test

def load_ndarray_label(name):
    """ Load the train,valid,test data for the dataset `name`
        and return it in ndarray format.

        This is only available for the toy dataset ule.
    """
    assert name in ['ule']
    trname,vname,tename = [os.path.join(config.data_root(),
                                        'UTLC','filetensor',
                                        name+'_'+subset+'.ft')
                           for subset in ['trainl','validl','testl']]
    trainl = load_filetensor(trname)
    validl = load_filetensor(vname)
    testl = load_filetensor(tename)
    return trainl, validl, testl

def load_filetensor(fname):
    f = None
    try:
        if not os.path.exists(fname):
            fname = fname+'.gz'
            assert os.path.exists(fname)
            f = gzip.open(fname)
        elif fname.endswith('.gz'):
            f = gzip.open(fname)
        else:
            f = open(fname)
        d = ft.read(f)
    finally:
        if f:
            f.close()

    return d

def load_sparse(fname):
    f = None
    try:
        if not os.path.exists(fname):
            fname = fname+'.gz'
            assert os.path.exists(fname)
            f = gzip.open(fname)
        elif fname.endswith('.gz'):
            f = gzip.open(fname)
        else:
            f = open(fname)
        d = cPickle.load(f)
    finally:
        if f:
            f.close()
    return d

if __name__ == '__main__':
    import numpy
    import scipy.sparse

    # Test loading of transfer data
    train, valid, test, transfer = load_ndarray_dataset("ule", normalize=True, transfer=True)
    assert train.shape[0]==transfer.shape[0]

    for name in ['avicenna','harry','rita','sylvester','ule']:
        train, valid, test = load_ndarray_dataset(name, normalize=True)
        print name,"dtype, max, min, mean, std"
        print train.dtype, train.max(), train.min(), train.mean(), train.std()
        assert isinstance(train, numpy.ndarray)
        assert isinstance(valid, numpy.ndarray)
        assert isinstance(test, numpy.ndarray)
        assert train.shape[1]==test.shape[1]==valid.shape[1]

    # Test loading of transfer data
    train, valid, test, transfer = load_sparse_dataset("ule", normalize=True, transfer=True)
    assert train.shape[0]==transfer.shape[0]

    for name in ['harry','terry','ule']:
        train, valid, test = load_sparse_dataset(name, normalize=True)
        nb_elem = numpy.prod(train.shape)
        mi = train.data.min()
        ma = train.data.max()
        mi = min(0, mi)
        ma = max(0, ma)
        su = train.data.sum()
        mean = float(su)/nb_elem
        print name,"dtype, max, min, mean, nb non-zero, nb element, %sparse"
        print train.dtype, ma, mi, mean, train.nnz, nb_elem, (nb_elem-float(train.nnz))/nb_elem
        print name,"max, min, mean, std (all stats on non-zero element)"
        print train.data.max(), train.data.min(), train.data.mean(), train.data.std()
        assert scipy.sparse.issparse(train)
        assert scipy.sparse.issparse(valid)
        assert scipy.sparse.issparse(test)
        assert train.shape[1]==test.shape[1]==valid.shape[1]
