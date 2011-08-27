import os
import numpy
from ..io.filetensor import read
from .config import data_root
from .dataset import Dataset

def load_file(info, normalize=True, mode='stereo', downsample_amt=1, dtype='float64'):
    """ Load the smallNorb data into numpy matrices.

    normalize_pixels True will divide the values by 255, which makes sense in conjunction
    with dtype=float32 or dtype=float64.

    """
    assert mode in ('stereo','mono')
    # NotImplementedError: subtensor access not written yet
    #subt = [numpy.arange(self.dim[0]), 
            #numpy.arange(0,self.dim[1],downsample_amt),
            #numpy.arange(0,self.dim[2],downsample_amt)]

    dat = read(open(info['dat']))
    if downsample_amt != 1:
        dat = dat[:, :, ::downsample_amt, ::downsample_amt]
    if dtype != 'int8':
        dat = numpy.asarray(dat, dtype=dtype)
    if normalize:
        dat  *= (1.0 / 255.0)
    if mode == 'mono':
        dat = dat[:,0,:,:]

    labels  = read(open(info['cat']))

    return dat, labels


#TODO
class NORB_small(object):
    """
    This class loads the original small NORB dataset.
    see http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/ for details.

    smallNORB is a rather large dataset. As such training and testing datasets
    are loaded dynamically. On the first access to .train member, the
    training dataset is loaded and split into training and validation set.
    Subsequent references to train and valid will not require any loading. The
    first subsequent access to .test member however will try to overwrite the 
    training dataset with the test dataset. This is achieved by deleting all 
    local references to the object. 

    To maintain this behaviour, it is important for the user not to maintain
    useless references fo the .train, .valid and .test members. Also be
    conscious of the penalty when alternating accesses between both datasets.

    TODO:
    James: like, as far as code design goes... how about writing those three functions,
    putting them into smallNORB in pylearn... then using those functions to implement the
    Trainer-compatible class in just a few lines?
    Guillaume:  yeah that's probably not a bad idea
    James: if you like the suggestion, you could also just paste it into a comment, and
    leave it for after NIPS :)
    Guillaume:  i like that one even more :)
    """

    class Paths():
        dirpath = os.path.join(data_root(), 'norb_small', 'original')
        train = {}
        test = {}
        train['dat'] = os.path.join(dirpath, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
        train['cat'] = os.path.join(dirpath, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
        train['info'] = os.path.join(dirpath, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat')
        test['dat']  = os.path.join(dirpath, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
        test['cat']  = os.path.join(dirpath, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')
        test['info']  = os.path.join(dirpath, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat')
    path = Paths()

    def __init__(self, ntrain=19440, nvalid=4860, ntest=24300,
               valid_variant=None,
               downsample_amt=1, seed=1, normalize=False,
               mode='stereo', dtype='int8'):

        self.n_classes = 5
        self.nsamples = 24300
        self.img_shape = (2,96,96) if mode=='stereo' else (96,96)
        self.mode = mode

        self.ntrain = ntrain
        self.nvalid = nvalid
        self.ntest = ntest
        self.downsample_amt = downsample_amt
        self.normalize = normalize
        self.dtype = dtype

        rng = numpy.random.RandomState(seed)
        if valid_variant is None:
            # The validation set is just a random subset of training
            self.indices = rng.permutation(self.nsamples)
            self.itr  = self.indices[0:ntrain]
            self.ival = self.indices[ntrain:ntrain+nvalid]
        elif valid_variant in (4,6,7,8,9):
            # The validation set consists in an instance of each category
            # In order to know which indices correspond to which instance,
            # we need to load the 'info' files.
            train_info = read(open(self.path.train['info']))

            ordered_itrain = numpy.nonzero(train_info[:,0] != valid_variant)[0]
            max_ntrain = ordered_itrain.shape[0]
            ordered_ivalid = numpy.nonzero(train_info[:,0] == valid_variant)[0]
            max_nvalid = ordered_ivalid.shape[0]

            if self.ntrain > max_ntrain:
                print 'WARNING: ntrain is %i, but there are only %i training samples available' % (self.ntrain, max_ntrain)
                self.ntrain = max_ntrain

            if self.nvalid > max_nvalid:
                print 'WARNING: nvalid is %i, but there are only %i validation samples available' % (self.nvalid, max_nvalid)
                self.nvalid = max_nvalid

            # Randomize
            print 
            self.itr = ordered_itrain[rng.permutation(max_ntrain)][0:self.ntrain]
            self.ival = ordered_ivalid[rng.permutation(max_nvalid)][0:self.nvalid]

        self.current = None

    def preprocess(self, x):
        if not self.normalize:
            return numpy.float64(x *1.0 / 255.0)
        return x

    def load(self, dataset='train'):

        if dataset == 'train' or dataset=='valid':
            print 'accessing train or valid dataset'

            if self.current != 'train':
                if self.current: del self.dat1

                print 'need to reload from train file'
                dat, cat  = load_file(self.path.train, self.normalize,
                                      self.mode, self.downsample_amt, self.dtype)

                x = dat[self.itr,...].reshape(self.ntrain,-1)
                y = cat[self.itr]
                self.dat1 = Dataset.Obj(x=x, y=y) # training

                x = dat[self.ival,...].reshape(self.nvalid,-1)
                y = cat[self.ival]
                self.dat2 = Dataset.Obj(x=x, y=y) # validation

                del dat, cat, x, y

            rval = self.dat1 if dataset=='train' else self.dat2 
            self.current = 'train'

        elif dataset=='test':

            print 'retrieving test set'
            if self.current!='test':
                if self.current: del self.dat1, self.dat2

                print 'need to reload from test file'
                dat, cat = load_file(self.path.test, self.normalize,
                                     self.mode, self.downsample_amt, self.dtype)

                x = dat.reshape(self.nsamples,-1)
                y = cat
                self.dat1 = Dataset.Obj(x=x, y=y)

                del dat, cat, x, y

            rval = self.dat1
            self.current = 'test'
        else:
            raise ValueError("Expected one of [train|valid|test]")

        return rval

    def __getattribute__(self, name):
        if name in ('train','valid','test'):
            print 'hello'
            return object.__getattribute__(self, 'load')(name)
        else:
            return object.__getattribute__(self, name)
