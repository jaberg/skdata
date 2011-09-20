from skdata import larochelle_etal_2007 as L2007
from skdata import tasks

def dset(name):
    rval = getattr(L2007, name)()
    rval.DOWNLOAD_IF_MISSING = False
    return rval

def test_MnistBasic():
    dsetname = 'MNIST_Basic'

    aa = dset(dsetname)
    aa.DOWNLOAD_IF_MISSING = False
    assert aa.meta_const['image']['shape'] == (28, 28)
    assert aa.meta_const['image']['dtype'] == 'float32'
    assert aa.descr['n_classes'] == 10
    assert aa.meta[0] == dict(id=0, label=5, split='train')
    assert aa.meta[9999] == dict(id=9999, label=7, split='train')
    assert aa.meta[10000] == dict(id=10000, label=3, split='valid')
    assert aa.meta[11999] == dict(id=11999, label=3, split='valid')
    assert aa.meta[12000] == dict(id=12000, label=7, split='test')
    assert aa.meta[50000] == dict(id=50000, label=3, split='test')
    assert aa.meta[61989] == dict(id=61989, label=4, split='test')
    assert len(aa.meta) == 62000

    bb = dset(dsetname)
    assert bb.meta == aa.meta

def test_several():
    dsetnames = ['MNIST_Basic',
            'MNIST_BackgroundImages',
            'MNIST_BackgroundRandom',
            'Rectangles',
            'RectanglesImages',
            'Convex']
    dsetnames.extend(['MNIST_Noise%i' % i for i in range(1,7)])
    for dsetname in dsetnames:

        aa = dset(dsetname)
        assert len(aa.meta) == sum(
                [aa.descr[s] for s in 'n_train', 'n_valid', 'n_test'])

        bb = dset(dsetname)
        assert aa.meta == bb.meta

        tasks.assert_classification(*aa.classification_task())
        tasks.assert_latent_structure(aa.latent_structure_task())

