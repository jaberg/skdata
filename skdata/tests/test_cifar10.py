from skdata import cifar10, tasks

def test_CIFAR10():
    cifar = cifar10.CIFAR10()  # just make sure we can create the class
    cifar.DOWNLOAD_IF_MISSING = False
    assert cifar.meta_const['image']['shape'] == (32, 32, 3)
    assert cifar.meta_const['image']['dtype'] == 'uint8'
    assert cifar.descr['n_classes'] == 10
    assert cifar.meta[0] == dict(id=0, label='frog', split='train')
    assert cifar.meta[49999] == dict(id=49999, label='automobile', split='train')
    assert cifar.meta[50000] == dict(id=50000, label='cat', split='test')
    assert cifar.meta[59999] == dict(id=59999, label='horse', split='test')
    assert len(cifar.meta) == 60000


def test_classification():
    cifar = cifar10.CIFAR10()  # just make sure we can create the class
    cifar.DOWNLOAD_IF_MISSING = False
    X, y = cifar.classification_task()
    tasks.assert_classification(X, y, 60000)


def test_latent_structure():
    cifar = cifar10.CIFAR10()  # just make sure we can create the class
    cifar.DOWNLOAD_IF_MISSING = False
    X = cifar.latent_structure_task()
    tasks.assert_latent_structure(X, 60000)


def test_meta_cache():
    a = cifar10.CIFAR10()
    b = cifar10.CIFAR10()
    assert a.meta == b.meta
