import numpy as np

from skdata.posner_keele.dataset import level_of_distortion
from skdata.posner_keele.dataset import distort
from skdata.posner_keele.dataset import int_spiral
from skdata.posner_keele.dataset import prototype_coords
from skdata.posner_keele.dataset import render_coords

def test_lod():
    for key, value in level_of_distortion.items():
        assert np.allclose(np.sum(value), 1.0)


def test_spiral():
    s0 = int_spiral(0)
    assert s0 == [(0, 0)]

    s1 = int_spiral(11)
    assert s1 == [
        (0, 0), (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1), (2, -1), (2, 0)]

    s2 = int_spiral(400)
    assert s2[-4:] == [(-6, 10), (-7, 10), (-8, 10), (-9, 10)]


def test_distort():
    rng = np.random.RandomState(4)
    new_pts = distort([[0, 0], [2, 1], [1, 3]], '2', rng)
    assert np.allclose(new_pts, [(1, -1), (2, -1), (1, 3)])

    new_pts = distort([[0, 0], [2, 1], [1, 3]], '7.7', rng)
    assert np.allclose(new_pts, [(5, -4), (2, 2), (-5, -5)])


def test_boundary_conditions():
    rng = np.random.RandomState(4)
    acc = None
    for i in range(1000):
        coords = prototype_coords(rng)
        dcoords = distort(coords, '7.7', rng)
        assert dcoords.min() >= 0
        assert dcoords.max() < 50
        if acc is None:
            acc = render_coords(dcoords)
        else:
            acc += render_coords(dcoords)

    if 0:
        import matplotlib.pyplot as plt
        plt.imshow(acc, cmap='gray', interpolation='nearest')
        plt.show()




