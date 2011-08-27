"""
Load Tzanetakis' genre-classification dataset.

"""
from __future__ import absolute_import

import os
import numpy

from ..io.amat import AMat
from .config import data_root
from .dataset import dataset_factory, Dataset

def centre_data(x, inplace=False):
    rval = x if inplace else x.copy()
    #zero-mean
    rval -= numpy.mean(rval, axis=0)
    #unit-variance
    rval *= 1.0 / (1.0e-6 + numpy.std(rval, axis=0))
    return rval

def mfcc16(segments_per_song = 1, include_covariance = True, random_split = 0,
        ntrain = 700, nvalid = 100, ntest = 200,
        normalize=True):
    if segments_per_song != 1:
        raise NotImplementedError()

    path = os.path.join(data_root(), 'tzanetakis','feat_mfcc16_540_1.stat.amat')
    dat = AMat(path=path)
    all_input = dat.input
    assert all_input.shape == (1000 * segments_per_song, 152)
    all_targ = numpy.tile(numpy.arange(10).reshape(10,1), 100 * segments_per_song)\
            .reshape(1000 * segments_per_song)

    if not include_covariance:
        all_input = all_input[:,0:16] 

    #shuffle the data according to the random split
    assert all_input.shape[0] == all_targ.shape[0]
    seed = random_split + 1
    numpy.random.RandomState(seed).shuffle(all_input)
    numpy.random.RandomState(seed).shuffle(all_targ)

    #construct a dataset to return
    rval = Dataset()

    def prepx(x):
        return centre_data(x, inplace=True) if normalize else x

    rval.train = Dataset.Obj(x=prepx(all_input[0:ntrain]),
            y=all_targ[0:ntrain])
    rval.valid = Dataset.Obj(x=prepx(all_input[ntrain:ntrain+nvalid]),
            y=all_targ[ntrain:ntrain+nvalid])
    rval.test =  Dataset.Obj(x=prepx(all_input[ntrain+nvalid:ntrain+nvalid+ntest]),
            y=all_targ[ntrain+nvalid:ntrain+nvalid+ntest])

    rval.n_classes = 10

    return rval

import theano

class TzanetakisExample(theano.Op):
    """Return the i'th file, label pair from the Tzanetakis dataset."""
    @staticmethod
    def read_tracklist(alt_path_root=None):
        """Read the tzanetakis dataset file 
        :rtype: (list, list)
        :returns: paths, labels
        """
        tracklist = open(data_root() + '/tzanetakis/tracklist.txt')
        path = []
        label = []
        for line in tracklist:
            toks = line.split()
            try:
                if alt_path_root is None:
                    path.append(toks[0])
                else:
                    line_path = toks[0]
                    file_name = line_path.split('/')[-1] 
                    path.append(alt_path_root + '/' + file_name)
                label.append(toks[1])
            except:
                print 'BAD LINE IN TZANETAKIS TRACKLIST'
                print line, toks
                raise
        assert len(path) == 1000
        return path, label

    class_idx_dict = dict(blues=numpy.asarray(0),
            classical=1,
            country=2,
            disco=3,
            hiphop=4,
            jazz=5,
            metal=6,
            pop=7,
            reggae=8,
            rock=9)
            
    def __init__(self, alt_path_root=None):
        self.path, self.label = self.read_tracklist(alt_path_root)
        self.class_idx_dict = {}
        classes = ('blues classical country disco hiphop jazz metal pop reggae rock').split()
        for i, c in enumerate(classes):
            self.class_idx_dict[c] = numpy.asarray(i, dtype='int64')

    n_examples = property(lambda self: len(self.path))
    nclasses = property(lambda self: 10)


    def make_node(self, idx):
        idx_ = theano.tensor.as_tensor_variable(idx)
        if idx_.type not in theano.tensor.int_types:
            raise TypeError(idx)
        return theano.Apply(self, 
                [idx_],
                [theano.generic('tzanetakis_path'), 
                    theano.tensor.lscalar('tzanetakis_label')])

    def perform(self, node, (idx,), (path, label)):
        path[0] = self.path[idx]
        label[0] = self.class_idx_dict[self.label[idx]]

    def grad(self, inputs, g_output):
        return [None for i in inputs]

#tzanetakis_example = TzanetakisExample() #requires reading a data file

