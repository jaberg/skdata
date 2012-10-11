import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from scipy.misc import fromimage
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

    def f_map(self, file_paths):
        if isinstance(file_paths, str):
            raise TypeError(file_paths)
        if self._shape:
            rval = np.empty((len(file_paths),) + self._shape, dtype='uint8')
        else:
            rval = [None] * len(file_paths)
        for ii, file_path in enumerate(file_paths):
            im_ii = imread(file_path, mode=self.mode)
            if len(im_ii.shape) not in (2, 3):
                raise IOError('Failed to decode %s' % file_path)
            img_ii = np.asarray(im_ii, dtype='uint8')
            assert len(img_ii.shape) in (2, 3)
            # -- broadcast pixels over channels if channels have been
            #    requested (_shape has len 3) and are not present
            #    (img_ii.ndim == 2)
            if img_ii.ndim == 2 and rval.ndim == 4:
                rval[ii] =  img_ii[:, :, np.newaxis]
            else:
                rval[ii] =  img_ii
        rval = rval.astype(self._dtype)
        if 'float' in str(self._dtype):
            rval /= 255.0
        return rval

    def __call__(self, file_path):
        return self.f_map([file_path])[0]


# XXX: these loaders currently do not coerce the loaded images
#      to be e.g. rgb or bw. Should they?
load_rgb_f32 = ImgLoader(ndim=3, dtype='float32')
load_rgb_u8 = ImgLoader(ndim=3, dtype='uint8')
load_bw_f32 = ImgLoader(ndim=2, dtype='float32')
load_bw_u8 = ImgLoader(ndim=2, dtype='uint8')

