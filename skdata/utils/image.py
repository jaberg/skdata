import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from scipy.misc import imresize, fromimage
except ImportError:
    logger.warn("The Python Imaging Library (PIL)"
            " is required to load data from jpeg files.")


def imread(name, flatten=0, mode=None):
    im = Image.open(name)
    if mode is not None and im.mode != mode:
        im = im.convert(mode)
    return fromimage(im, flatten=flatten)


class ImgLoader(object):
    def __init__(self, shape=None, ndim=None, dtype='uint8', mode=None):
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.mode = mode

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        rval = np.asarray(imread(file_path, mode=self.mode),
                          dtype=self._dtype)
        if 'float' in str(self._dtype):
            rval /= 255.0
        if self._ndim is not None and self._ndim != rval.ndim:
            raise ValueError('ndim', (self._ndim, rval.ndim))
        if self._shape is not None:
            assert self._ndim is not None
            for s0, s1 in zip(self._shape, rval.shape):
                if s0 is not None and s0 != s1:
                    raise ValueError('shape', (self._shape, rval.shape))
        return rval

# XXX: these loaders currently do not coerce the loaded images
#      to be e.g. rgb or bw. Should they?
load_rgb_f32 = ImgLoader(ndim=3, dtype='float32')
load_rgb_u8 = ImgLoader(ndim=3, dtype='uint8')
load_bw_f32 = ImgLoader(ndim=2, dtype='float32')
load_bw_u8 = ImgLoader(ndim=2, dtype='uint8')

if 0:
    def load_lfw_pairs(subset='train',
                       data_home=None,
                       funneled=True,
                       resize=0.5,
                       color=False,
                       slice_=(slice(70, 195), slice(78, 172)),
                       download_if_missing=False):
        """Loader for the Labeled Faces in the Wild (LFW) pairs dataset

        This dataset is a collection of JPEG pictures of famous people
        collected on the internet, all details are available on the
        official website:

            http://vis-www.cs.umass.edu/lfw/

        Each picture is centered on a single face. Each pixel of each channel
        (color in RGB) is encoded by a float in range 0.0 - 1.0.

        The task is called Face Verification: given a pair of two pictures,
        a binary classifier must predict whether the two images are from
        the same person.

        In the official `README.txt`_ this task is described as the
        "Restricted" task.  The "Unrestricted" variant is not currently
        supported.

          .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

        Parameters
        ----------
        subset: optional, default: 'train'
            Select the dataset to load: 'train' for the development training
            set, 'test' for the development test set, and '10_folds' for the
            official evaluation set that is meant to be used with a 10-folds
            cross validation.

        data_home: optional, default: None
            Specify another download and cache folder for the datasets. By
            default all scikit learn data is stored in '~/scikit_learn_data'
            subfolders.

        funneled: boolean, optional, default: True
            Download and use the funneled variant of the dataset.

        resize: float, optional, default 0.5
            Ratio used to resize the each face picture.

        color: boolean, optional, default False
            Keep the 3 RGB channels instead of averaging them to a single
            gray level channel. If color is True the shape of the data has
            one more dimension than than the shape with color = False.

        slice_: optional
            Provide a custom 2D slice (height, width) to extract the
            'interesting' part of the jpeg files and avoid use statistical
            correlation from the background

        download_if_missing: optional, True by default
            If False, raise a IOError if the data is not locally available
            instead of trying to download the data from the source site.
        """

    #
    # Drivers for skdata/bin executables
    #

    def main_show():
        from glviewer import glumpy_viewer, command, glumpy
        try:
            import argparse   # new in Python 2.7
            assert sys.argv[1] == 'lfw'
            sys.argv[1:2] = []

            parser = argparse.ArgumentParser(
                    description='Show the Labeled Faces in the Wild (lfw) dataset')
            # task
            parser.add_argument('task',
                    type=str,
                    default='people',
                    help='task: "pairs" or "people"')
            # color
            parser.add_argument('--color', action='store_true', dest='color',
                    help='load the images in color (default)')
            parser.add_argument('--no-color', action='store_false', dest='color')
            # resize
            parser.add_argument('--resize', type=float, default=1.0,
                    help="fraction of original image size")
            # subset
            parser.add_argument('--subset', type=str, default='train',
                    help='for "pairs", which subset to load (train/test/10_folds)')

            args = parser.parse_args()
            if args.task == 'people':
                people = load_lfw_people(
                        resize=args.resize,
                        color=args.color,
                        slice_=None)
                n_rows = len(people.imgs)
                print 'n. rows', n_rows
                glumpy_viewer(
                        img_array=people.imgs,
                        arrays_to_print=[people.target, people.names],
                        cmap=glumpy.colormap.Grey)
            elif args.task == 'pairs':
                pairs = load_lfw_pairs(
                        subset=args.subset,
                        resize=args.resize,
                        color=args.color,
                        slice_=None)
                n_rows = len(pairs.left_imgs)
                print 'n. rows', n_rows
                glumpy_viewer(
                        img_array=pairs.left_right_imgs,
                        arrays_to_print=[pairs.target, pairs.names],
                        cmap=glumpy.colormap.Grey,
                        window_shape=(512, 256))

            else:
                raise NotImplementedError(args.task)
        except ImportError:
            logger.warn('no argparse - ignoring arguments')
            # argparse isn't installed, so just show something
            people = load_lfw_people()
            n_rows = len(people.imgs)
            print 'n. rows', n_rows
            glumpy_viewer(
                    img_array=people.imgs,
                    arrays_to_print=[people.target, people.names])
