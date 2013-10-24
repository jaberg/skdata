import sys
import logging
import time

from skdata.kaggle_facial_expression.dataset \
        import KaggleFacialExpression, FULL_URL

usage = """
Usage: main.py {bib, clean_up, install, show, stats}
"""

def main_install():
    """
    Download the CIFAR10 data set to the skdata cache dir
    """
    try:
        KaggleFacialExpression().install(sys.argv[2])
    except IndexError:
        print "To download the Facial Recognition Challenge dataset"
        print "log into Kaggle and retrieve", FULL_URL
        print "then run this script again with the downloaded filename"
        print "as an argument, like:"
        print "python -m %s.main install fer2013.tgz" % __package__


def main_bib():
    """
    Print out the proper citation for this dataset.
    """
    print open(KaggleFacialExpression().home('fer2013', 'fer2013.bib')).read()


def main_stats():
    """
    Print some basic stats about the dataset (proving that it can be loaded).
    """
    self = KaggleFacialExpression()
    t0 = time.time()
    print 'loading dataset ...'
    meta = self.meta
    print ' ... done. (%.2f seconds)' % (time.time() - t0)
    print ''
    print 'n. Examples', len(meta)
    #print 'Usages', set([m['usage'] for m in meta])
    print 'n. Training', len([m for m in meta if m['usage'] == 'Training'])
    print 'n. PublicTest', len([m for m in meta if m['usage'] == 'PublicTest'])
    print 'n. PrivateTest', len([m for m in meta if m['usage'] == 'PrivateTest'])


def main_show():
    """
    Use glumpy to launch a data set viewer.
    """
    self = KaggleFacialExpression()
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


