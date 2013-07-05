import sys
import logging

from skdata.kaggle_facial_expression.dataset \
        import KaggleFacialExpression

usage = """
Usage: main.py {fetch, show, clean_up}
"""

def main_fetch():
    """
    Download the CIFAR10 data set to the skdata cache dir
    """
    KaggleFacialExpression().fetch(True)


def main_show():
    """
    Use glumpy to launch a data set viewer.
    """
    self = KaggleFacialExpression()
    meta = self.meta
    from skdata.utils.glviewer import glumpy_viewer
    glumpy_viewer(
            img_array=[m['pixels'] for m in self.meta],
            arrays_to_print=self.meta,
            window_shape=(48 * 4, 48 * 4))


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


