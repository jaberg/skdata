from skdata import mnist, tasks

def test_MNIST():
    M = mnist.MNIST()  # just make sure we can create the class
    M.DOWNLOAD_IF_MISSING = False
    assert M.meta_const['image']['shape'] == (28, 28)
    assert M.meta_const['image']['dtype'] == 'uint8'
    assert M.descr['n_classes'] == 10
    assert M.meta[0] == dict(id=0, split='train', label=5), M.meta[0]
    assert M.meta[69999] == dict(id=69999, split='test', label=6), M.meta[69999]
    assert len(M.meta) == 70000


def test_MNIST_classification():
    M = mnist.MNIST()  # just make sure we can create the class
    M.DOWNLOAD_IF_MISSING = False
    X, y = M.classification_task()
    tasks.assert_classification(X, y, 70000)


def test_MNIST_latent_structure():
    M = mnist.MNIST()  # just make sure we can create the class
    M.DOWNLOAD_IF_MISSING = False
    X = M.latent_structure_task()
    tasks.assert_latent_structure(X, 70000)
