import numpy as np
from datasets import larray


def test_usage():
    np.random.seed(123)

    def load_rgb(pth):
        return pth + '_rgb'
    def load_grey(pth):
        return pth + '_grey'
    def to_64x64(img):
        return img + '_64x64'

    paths = ['a', 'b', 'c', 'd']  # imagine some huge list of image paths
    rgb_imgs = larray.lmap(load_rgb, paths)
    grey_imgs = larray.lmap(load_grey, paths)
    paths_64x64 = larray.lmap(to_64x64, grey_imgs)

    train_set = larray.reindex(rgb_imgs, np.random.permutation(len(paths))).loop()

    l10 = list(train_set[range(10)])
    print l10
    assert ['d', 'a', 'b', 'c'] == [l[0] for l in l10[:4]]


def test_using_precompute():
    np.random.seed(123)

    # example library code  starts here
    def load_rgb(pth):
        return pth + '_rgb'
    def load_grey(pth):
        return pth + '_grey'
    def to_64x64(img):
        return img + '_64x64'

    paths = ['a', 'b', 'c', 'd']  # imagine some huge list of image paths
    rgb_imgs = larray.lmap(load_rgb, paths)
    grey_imgs = larray.lmap(load_grey, paths)
    paths_64x64 = larray.lmap(to_64x64, grey_imgs)

    train_set = larray.reindex(paths_64x64, np.random.permutation(len(paths))).loop()

    # example user code starts here.
    # It is easy to memmap the __array__ of paths_64x64, but
    # it is more difficult to compute derived things using that
    # memmap.
    
    # pretend this is a memmap of a precomputed quantity, for example.
    use_paths_64x64 = ['stuff', 'i', 'saved', 'from', 'disk']

    # the rest of the original graph (e.g. train_set)
    # doesn't know about our new memmap
    # or mongo-backed proxy, or whatever we're doing.

    new_train_set = larray.clone(train_set, given={paths_64x64: use_paths_64x64})

    l10 = list(new_train_set[range(10)])
    print l10
    assert l10 == [
            'from', 'stuff', 'i', 'saved',
            'from', 'stuff', 'i', 'saved',
            'from', 'stuff']


def test_pprint():
    paths = None
    rgb_imgs = larray.lmap(test_pprint, paths)
    rgb_imgs2 = larray.lmap(test_pprint, rgb_imgs)
    s = larray.pprint_str(rgb_imgs2)
    print s
    assert s == """lmap(test_pprint, ...)
    lmap(test_pprint, ...)
        None\n"""
