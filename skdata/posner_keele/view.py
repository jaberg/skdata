import copy
import functools

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

from skdata.base import Task

from dataset import prototype_coords
from dataset import distort
from dataset import render_coords

def render_coords_uint8_channels(coords):
    n_points, n_dims = coords.shape
    assert n_dims == 2
    rval = render_coords(coords)
    rval *= 255
    rval = rval.astype('uint8')
    rval = rval[:, :, np.newaxis]
    return rval


def blur(self, X):
    rval = np.empty(X.shape)
    for i, Xi in enumerate(X):
        rval[i] = gaussian_filter(X[i].astype('float') / 255,
                                  sigma=self.blur_sigma,
                                  mode='constant')

    ### downsample
    down_size = (11,11)
    rval = rval[:,:,:,0]

    X2 = []
    for x in rval:
        X2.append( imresize(x, down_size) )
    rval = np.array(X2, dtype='float64') / 255.0

    return rval



class PosnerKeele1968E3(object):
    """

    Protocol of Experiment 3 from Posner and Keele, 1968.
    "On the Genesis of Abstract Ideas"

    """
    def __init__(self, seed=1, train_level='7.7'):
        self.seed = seed
        self.train_level = train_level
        self.n_prototypes = 3
        self.n_train_per_prototype = 4
        self.n_test_5_per_prototype = 2
        self.n_test_7_per_prototype = 2

    def distortion_set(self, N, coords, level, rng):
        images = []
        labels = []
        assert len(coords) == self.n_prototypes
        for proto, coord in enumerate(coords):
            # --apply the same distortions to each example
            rng_copy = copy.deepcopy(rng)
            for i in range(N):
                dcoord = distort(coord, level, rng_copy)
                img = render_coords_uint8_channels(dcoord)
                images.append(img)
                labels.append(proto)
        rng.seed(int(rng_copy.randint(2**30)))
        return np.asarray(images), np.asarray(labels)

    def task(self, name, images, labels):
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[:, :, :, np.newaxis]
        return Task(
            'indexed_image_classification',
            name=name,
            idxs=range(len(images)),
            all_images=images,
            all_labels=np.asarray(labels),
            n_classes=self.n_prototypes)

    def protocol(self, algo):
        rng = np.random.RandomState(self.seed)
        n_prototypes = self.n_prototypes

        coords = [prototype_coords(rng) for i in range(n_prototypes)]

        dset = functools.partial(self.distortion_set,
                                 coords=coords,
                                 rng=rng,
                                )

        train_images, train_labels = dset(
            N=self.n_train_per_prototype,
            level=self.train_level)

        test_5_images, test_5_labels = dset(
            N=self.n_test_5_per_prototype,
            level='5')
        test_7_images, test_7_labels = dset(
            N=self.n_test_7_per_prototype,
            level='7.7')

        test_proto_images, test_proto_labels = dset(N=1, level='0')
        test_proto_labels = range(self.n_prototypes)

        # XXX: Careful not to actually expect the model to get these right.
        test_random_images = [
            render_coords_uint8_channels(prototype_coords(rng))
            for c in range(n_prototypes)]
        test_random_labels = range(self.n_prototypes)

        model = algo.best_model(
            train=self.task('train', train_images, train_labels))


        loss_5 = algo.loss(model,
             self.task('test_5', test_5_images, test_5_labels))
        loss_7 = algo.loss(model,
             self.task('test_7', test_7_images, test_7_labels))
        loss_train = algo.loss(model,
             self.task('test_train', train_images, train_labels))
        loss_proto = algo.loss(model,
             self.task('test_proto', test_proto_images, test_proto_labels))
        loss_random = algo.loss(model,
             self.task('test_random', test_random_images, test_random_labels))

        return algo

