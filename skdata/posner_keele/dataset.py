"""
The data set implemented here is a synthetic visual data set of random dot
patterns, introduced by two fundamental psychological experiments for the study
pattern representation.  Subjects are trained to distinguish
3 broadly-different random patterns by inferring a rule from labeled
distortions of 3 prototypes.

Posner, M. I., & Keele, S. W. (1968).  On the genesis of abstract ideas.
Journal of experimental psychology, 77(3p1), 353.

Posner, M. I., Goldsmith, R., & Welton Jr, K. E. (1967). Perceived distance
and the classification of distorted patterns. Journal of Experimental
Psychology, 73(1), 28.

"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter

level_of_distortion = {
    '0': [1.0, 0, 0, 0, 0],
    '1': [.88, .1, .015, .004, .001],
    '2': [.75, .15, .05, .03, .02],
    '3': [.59, .20, .16, .03, .02],
    '4': [.36, .48, .06, .05, .05],
    '5': [.2, .3, .4, .05, .05],
    '6': [.0, .4, .32, .15, .13],
    '7.7': [.0, .24, .16, .3, .3],
}

# these are (low, high) randint ranges into the
# locations enumerated below in spiral400
adjacency_areas = [
    (0, 1),
    (1, 9),
    (9, 25),
    (25, 100),
    (100, 400)]


def int_spiral(N):
    """
    Return a list of 2d locations forming a spiral
    (0, 0),
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
    (2, -1), (2, 0), ...
    """

    def cw(a, b):
        if (a, b) == (1, 0):
            return (0, 1)
        elif (a, b) == (0, 1):
            return (-1, 0)
        elif (a, b) == (-1, 0):
            return (0, -1)
        else:
            return (1, 0)

    rval = []
    seen = set()
    rval.append((0, 0))
    seen.add((0, 0))

    i, j = 1, 0
    ti, tj = -1, 0
    di, dj = 0, 1

    while len(rval) < N:
        assert (i, j) not in seen
        rval.append((i, j))
        seen.add((i, j))
        if (i + ti, j + tj) in seen:
            i += di
            j += dj
        else: # -- turn a corner
            ti, tj = cw(ti, tj)
            di, dj = cw(di, dj)
            i += di
            j += dj
    return rval

spiral400 = int_spiral(400)


def distort(rowcols, level, rng):
    """
    Apply the distortion algorithm described in (Posner et al. 1967).
    """
    N = len(rowcols)
    rval = []
    if level in level_of_distortion:
        pvals = level_of_distortion[level]
        areas = rng.multinomial(n=1, pvals=pvals, size=(N,)).argmax(axis=1)
        assert len(rowcols) == len(areas)
        for (r, c), area_i in zip(rowcols, areas):
            pos = rng.randint(*adjacency_areas[area_i])
            dr, dc = spiral400[pos]
            rval.append((r + dr, c + dc))
    elif level == '8.6':
        for (r, c) in rowcols:
            pos = rng.randint(400)
            dr, dc = spiral400[pos]
            rval.append((r + dr, c + dc))
    elif level == '9.7':
        for (r, c) in rowcols:
            r, c = rng.randint(10, 40, size=(2,))
            rval.append((r, c))
    return np.asarray(rval)


def prototype_coords(rng):
    """
    Sample 2-d coordinates for a Posner-Keele trial.

    Returns a 9x2 matrix of point locations within a 50x50 grid. Points all
    lie within the 30x30 region at the centre of the 50x50 grid.

    """
    return rng.randint(10, 40, size=(9, 2))


def render_coords(coords, blur=True, blur_sigma=1.5, crop_30=True):
    """
    Render point coordinates into a two-dimensional image matrix.

    Returns: a 50x50 rendering (lossless) or a 30x30 crop from the centre if
    `crop_30` is True.
    """
    rval = np.zeros((50, 50))
    rval[coords[:, 0], coords[:, 1]] = 1

    if blur:
        # rval = gaussian_filter(rval, sigma=1.0, mode='constant')
        rval = gaussian_filter(rval, sigma=blur_sigma, mode='constant')
        rval = rval / rval.max()

        if crop_30:
            maxval = 0.8
            return rval[10:40, 10:40].clip(0,maxval) / maxval
        else:
            return rval
    else:

        if crop_30:
            return rval[10:40, 10:40]
        else:
            return rval




