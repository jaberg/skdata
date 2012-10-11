#!/usr/bin/env python

import sys
import logging


#import dataset
import view

from skdata.utils.glviewer import glumpy_viewer, glumpy

logger = logging.getLogger(__name__)

usage = """
Usage: main.py show <variant>

    <variant> can be one of {original, aligned, funneled}
"""


def main_show():
    """
    Use glumpy to launch a data set viewer.
    """
    variant = sys.argv[2]
    if variant == 'original':
        obj = view.Original()
        cmap=None
    elif variant == 'aligned':
        obj = view.Aligned()
        cmap=glumpy.colormap.Grey
    elif variant == 'funneled':
        obj = view.Funneled()
        cmap=None
    else:
        raise ValueError(variant)

    glumpy_viewer(
        img_array=obj.image_pixels,
        arrays_to_print=[obj.image_pixels],
        cmap=cmap,
        window_shape=(250, 250),
        )


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    if len(sys.argv) <= 1:
        print usage
        return 1
    else:
        try:
            fn = globals()['main_' + sys.argv[1]]
        except:
            print 'command %s not recognized' % sys.argv[1]
            print usage
            return 1
        return fn()


if __name__ == '__main__':
    sys.exit(main())

