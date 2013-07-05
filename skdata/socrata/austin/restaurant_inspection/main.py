"""
Commands for the restaurant_inspection

"""
import sys
import logging
import numpy as np

from sklearn.svm import LinearSVC
from skdata.base import SklearnClassifier

from skdata.socrata.austin.restaurant_inspection.dataset \
        import RestaurantInspectionScores

logger = logging.getLogger(__name__)

usage = """
Usage: main.py {print, hist}
"""

def main_print():
    ris = RestaurantInspectionScores()
    for dct in ris.meta:
        print dct


def main_hist():
    import matplotlib.pyplot as plt
    ris = RestaurantInspectionScores()
    scores = [float(dct['score']) for dct in ris.meta]
    print scores
    plt.hist(scores)
    plt.xlabel('Inspection Score')
    plt.ylabel('Frequency')
    plt.title('Restaurant Inspection Scores: Austin, TX')
    plt.show()


def main_coord_scatter():
    import matplotlib.pyplot as plt
    ris = RestaurantInspectionScores()
    scores = [dct['score'] for dct in ris.meta]
    latitudes = [dct['address']['latitude'] for dct in ris.meta]
    longitudes = [dct['address']['longitude'] for dct in ris.meta]
    c = ((np.asarray(scores) - 50) / 50.0)[:, None] + [0, 0, 0]

    plt.scatter(latitudes, longitudes, c=c)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Restaurant Inspection Scores By Location: Austin, TX')
    plt.show()


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

