"""
A few helpful scripts specific to the CIFAR10 data set.

"""
import sys
import logging

from skdata.cifar10.dataset import CIFAR10

usage = """
Usage: main.py {fetch, show, clean_up}
"""


def main_fetch():
    """
    Download the CIFAR10 data set to the skdata cache dir
    """
    CIFAR10().fetch(True)


def main_show():
    """
    Use glumpy to launch a data set viewer.
    """
    from skdata.utils.glviewer import glumpy_viewer
    self = CIFAR10()
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=self._pixels,
            arrays_to_print=[Y],
            window_shape=(32 * 4, 32 * 4))


def main_clean_up():
    """
    Delete all memmaps and data set files related to CIFAR10.
    """
    CIFAR10().clean_up()


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

