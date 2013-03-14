from skdata.mnist import dataset, view
# XXX It appears that these tests are *way* out of date. They use use the old
# interface, whereas MNIST uses the view interface now.

def test_MNIST():
    M = dataset.MNIST()  # just make sure we can create the class
    M.DOWNLOAD_IF_MISSING = False
    assert M.meta_const['image']['shape'] == (28, 28)
    assert M.meta_const['image']['dtype'] == 'uint8'
    assert M.descr['n_classes'] == 10
    assert M.meta[0] == dict(id=0, split='train', label=5), M.meta[0]
    assert M.meta[69999] == dict(id=69999, split='test', label=6), M.meta[69999]
    assert len(M.meta) == 70000


