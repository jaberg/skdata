import os
import numpy
from pylearn.io.filetensor import read
from pylearn.datasets.config import data_root

#Path = '/u/bergstrj/pub/data/smallnorb'
#Path = '/data/lisa/datasmallnorb'
#Path = '/home/louradou/data/norb'

class Paths(object):
    """File-related operations on smallNorb
    """
    def __init__(self):
        smallnorb = [data_root(), 'smallnorb']
        self.train_dat = os.path.join(*\
                smallnorb + ['smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat'])
        self.test_dat = os.path.join(*\
                smallnorb + ['smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat'])
        self.train_cat = os.path.join(*\
                smallnorb + ['smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'])
        self.test_cat = os.path.join(*\
                smallnorb + ['smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'])
        self.train_info = os.path.join(*\
                smallnorb + ['smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'])
        self.test_info = os.path.join(*\
                smallnorb + ['smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'])

    def load_append_train_test(self, normalize_pixels=True, downsample_amt=1, dtype='uint8'):
        """ Load the smallNorb data into numpy matrices.

        normalize_pixels True will divide the values by 255, which makes sense in conjunction
        with dtype=float32 or dtype=float64.

        """
        def downsample(dataset):
            return dataset[:, 0, ::downsample_amt, ::downsample_amt]

        samples = downsample(read(open(self.train_dat)))
        samples = numpy.vstack((samples, downsample(read(open(self.test_dat)))))
        samples = numpy.asarray(samples, dtype=dtype)
        if normalize_pixels:
            samples *= (1.0 / 255.0)

        labels = read(open(self.train_cat))
        labels = numpy.hstack((labels, read(open(self.test_cat))))

        infos = read(open(self.train_info))
        infos = numpy.vstack((infos, read(open(self.test_info))))

        return samples, labels, infos
    
def smallnorb_iid(ntrain=29160, nvalid=9720, ntest=9720, dtype='float64', normalize_pixels=True):
    """Variation of the smallNorb task in which we randomly shuffle all the object instances
    together before dividing into train/valid/test.

    The default train/valid/test sizes correspond to 60/20/20 split of the entire dataset.

    :returns: 5, (train_x, train_labels), (valid_x, valid_labels), (test_x, test_labels) 

    """
    # cut from /u/louradoj/theano/hpu/expcode1.py
    rng = numpy.random.RandomState(1)        
    samples, labels, infos = Paths().load_append_train_test(downsample_amt=3, dtype=dtype, normalize_pixels=normalize_pixels)

    nsamples = samples.shape[0]
    if ntrain + nvalid + ntest > nsamples:
        raise Exception("ntrain+nvalid+ntest exceeds number of samples (%i)" % nsamples, 
                (ntrain, nvalid, ntest))
    i0 = 0
    i1 = ntrain
    i2 = ntrain + nvalid
    i3 = ntrain + nvalid + ntest

    indices = rng.permutation(nsamples)
    train_rows = indices[i0:i1]
    valid_rows = indices[i1:i2]
    test_rows = indices[i2:i3]

    n_labels = 5

    def _pick_rows(rows):
        a = numpy.array([samples[i].flatten() for i in rows])
        b = numpy.array([labels[i] for i in rows])
        return a, b

    return [_pick_rows(r) for r in (train_rows, valid_rows, test_rows)]

def smallnorb_azSplit():
    # cut from /u/louradoj/theano/hpu/expcode1.py
    # WARNING NOT NECESSARILY WORKING CODE

    samples, labels, infos = _load_append_train_test()
    train_rows, valid_rows, test_rows = [], [], []
    train_rows_azimuth = []
    for instance in range(10):
        az_min = 4*instance
        az_max = 4*instance + 18
        train_rows_azimuth.append( [a % 36 for a in range(az_min,az_max,2)] )
    #print "train_rows_azimuth", train_rows_azimuth
    for i, info in enumerate(infos):
        if info[2] in train_rows_azimuth[info[0]]:
            train_rows.append(i)
        elif info[2] / 2 % 2 == 0:
            test_rows.append(i)
        else:
            valid_rows.append(i)

    return [_pick_rows(samples, labels, r) for r in (train_rows, valid_rows, test_rows)]
