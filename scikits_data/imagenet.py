import sys, time

from ..io.filetensor import arraylike
from .config import data_root

_filenames = [
    'n01627424.ft', #short compared to other ones (50K examples)
    'n01861778.ft',
    'n03405725.ft',
    'n04451818.ft',
    #'n09287968.ft', #short compared to other ones (6K examples)
    'n13134947.ft',
    'n01503061.ft',
    'n01661091.ft',
    'n02512053.ft',
    'n03800933.ft',
    'n04524313.ft',
    'n11669921.ft']

def _arraylike_list():
    return [arraylike(
                open(data_root()+'/image_net/filetensor/12_top_classes/'+f), 
                rank=1)
            for f in _filenames]

class _loop_range(object):
    def __init__(self, a, start=0, stop=None):
        self.a = a
        self.start = start
        self.stop = len(a) if stop is None else stop
        self.len = self.stop - self.start
        if self.len <= 0:
            raise ValueError('len must be positive')

    def __getitem__(self, i):
        return self.a[self.start + i % self.len]

def example_stream_train(test_per_class=10000):
    streams = [_loop_range(a, start=test_per_class)
            for a in _arraylike_list()]
    poslist = [0 for s in streams]

    label = 0
    while True:
        yield streams[label][poslist[label]], label
        poslist[label] += 1
        label = (label + 1) % len(streams)
    
def example_stream_test(test_per_class=10000):
    for label, a in enumerate(_arraylike_list()):
        #print 'class %3i'%label, 'size %8i'% len(a)
        for i, a_i in enumerate(a):
            if i == test_per_class:
                break
            yield a_i, label
        if i != test_per_class:
            print >> sys.stderr, "Warning: class too short:", label, i



def test_example_stream_test():
    t = time.time()
    ntestperclass=10000
    for i, e in enumerate(example_stream_test(ntestperclass)):
        #sys.stdout.write('.')
        pass
    print time.time()-t
    assert i+1 == len(_filenames)*ntestperclass

def test_example_stream_train():
    t=time.time()
    for i, (data, label) in enumerate(example_stream_train(10000)):
        #print i, label, data.shape
        if i == len(_filenames)*50000:
            break
    print 'reading', i, 'examples took', time.time()-t
