import sys
import numpy as np

from dataset import prototype_coords
from dataset import render_coords
from dataset import distort
from view import PosnerKeele1968E3

import skdata

def main_show():
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(1)
    coords = prototype_coords(rng)
    img = render_coords(coords)
    img3 = render_coords(distort(coords, '3', rng))
    img6 = render_coords(distort(coords, '6', rng))
    plt.imshow(np.asarray([img, img3, img6]).T,
               cmap='gray',
               interpolation='nearest')
    plt.show()


def main_dump_to_png():
    from PIL import Image
    class DumpAlgo(skdata.base.LearningAlgo):
        def __init__(self, seed=123):
            self.rng = np.random.RandomState(seed)

        def forget(self, model):
            pass

        def best_model(self, train, valid=None):
            return getattr(self, 'best_model_' + train.semantics)(train, valid)

        def best_model_indexed_image_classification(self, train, valid):
            assert valid is None
            self.loss(None, train)

        def loss(self, model, task):
            for ii in task.idxs:
                filename = 'pk_%s_%i_label_%i.png' % (
                    task.name, ii, task.all_labels[ii])
                imga = task.all_images[ii][:, :, 0]
                print 'Saving', filename
                Image.fromarray(imga, 'L').save(filename)

    pk = PosnerKeele1968E3()
    pk.protocol(DumpAlgo())


if __name__ == '__main__':
    sys.exit(globals()['main_' + sys.argv[1]]())

