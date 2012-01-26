import tempfile
import numpy as np
from skdata import larray


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

    train_set = larray.reindex(rgb_imgs, np.random.permutation(len(paths))
                              ).loop()

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
    grey_imgs = larray.lmap(load_grey, paths)
    paths_64x64 = larray.lmap(to_64x64, grey_imgs)

    train_set = larray.reindex(paths_64x64, np.random.permutation(len(paths))
                              ).loop()

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


def test_lprint():
    paths = None
    rgb_imgs = larray.lmap(test_lprint, paths)
    rgb_imgs2 = larray.lmap(test_lprint, rgb_imgs)
    s = larray.lprint_str(rgb_imgs2)
    print s
    assert s == """lmap(test_lprint, ...)
    lmap(test_lprint, ...)
        None\n"""

larray.cache_memmap.ROOT = tempfile.mkdtemp(prefix="skdata_test_memmap_root")

class TestCache(object):
    def battery(self, cls):
        base0 = np.arange(10)
        base1 = -np.arange(10)
        base = np.vstack([base0, base1]).T
        # base[0] = [0, 0]
        # base[1] = [1, -1]
        # ...
        cpy = larray.lzip(base0, base1)
        cached = cls(cpy)
        assert cached.dtype == base.dtype
        assert cached.shape == base.shape
        def assert_np_eq(l, r):
            assert np.all(l == r), (l, r)
        assert_np_eq(cached._valid, 0)
        assert cached.rows_computed == 0
        assert_np_eq(cached[4], base[4])
        assert_np_eq(cached._valid, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert cached.rows_computed == 1
        assert_np_eq(cached[1], base[1])
        assert_np_eq(cached._valid, [0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        assert_np_eq(cached[0:5], base[0:5])
        n_computed = cached.rows_computed
        assert_np_eq(cached._valid, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test that asking for existing stuff doen't mess anything up
        # or compute any new rows
        assert_np_eq(cached[0:5], base[0:5])
        assert_np_eq(cached._valid, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert n_computed == cached.rows_computed

        # test that we can ask for things off the end
        assert_np_eq(cpy[8:16], base[8:16])
        assert_np_eq(cached[8:16], base[8:16])
        assert_np_eq(cached._valid, [1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

        cached.populate()
        assert np.all(cached._valid)
        assert_np_eq(cached._data, base)

    def test_memmap_cache(self):
        self.battery(lambda obj: larray.cache_memmap(obj, 'name_foo'))

    def test_memory_cache(self):
        self.battery(larray.cache_memory)
