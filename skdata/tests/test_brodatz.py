
from skdata.brodatz import *


def test_smoke():
    ds = Brodatz()
    assert len(ds.meta) == 111

    for m in ds.meta:
        assert (600, 600) < m['image']['shape'] < (700, 700), m

