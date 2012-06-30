"""
A helpful scripts specific to the MNIST data set.

"""
import sys
import logging


from skdata.mnist.dataset import MNIST
from skdata.utils.glviewer import glumpy_viewer, glumpy

logger = logging.getLogger(__name__)

usage = """
Usage: main.py {fetch, show, clean_up}
"""


def main_fetch():
    """
    Download the MNIST data set to the skdata cache dir
    """
    MNIST().fetch(download_if_missing=True)


def main_show():
    """
    Use glumpy to launch a data set viewer.
    """
    self = MNIST()
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=self.arrays['train_images'],
            arrays_to_print=[Y],
            cmap=glumpy.colormap.Grey,
            window_shape=(28 * 4, 28 * 4)
            )


def main_clean_up():
    """
    Delete all memmaps and data set files related to MNIST.
    """
    logger.setLevel(logging.INFO)
    MNIST().clean_up()



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

