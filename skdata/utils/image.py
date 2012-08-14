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
        if self._ndim is not None and self._ndim != rval.ndim:
            img = Image.open(file_path)
            # loading triggers a failure if a decoder is not available
            try:
                img.load()
            except:
                print ('HINT: To install decoders in PIL in virtualenv on'
                        ' ubuntu: '
                        'http://ubuntuforums.org/showthread.php?t=1751455')
                raise
            # otherwise let the ndim error through
            raise ValueError('ndim', (self._ndim, rval.ndim))
        if self._shape is not None:
            assert self._ndim is not None
            for s0, s1 in zip(self._shape, rval.shape):
                if s0 is not None and s0 != s1:
                    raise ValueError('shape', (self._shape, rval.shape))
        if 'float' in str(self._dtype):
            rval /= 255.0
        return rval


# XXX: these loaders currently do not coerce the loaded images
#      to be e.g. rgb or bw. Should they?
load_rgb_f32 = ImgLoader(ndim=3, dtype='float32')
load_rgb_u8 = ImgLoader(ndim=3, dtype='uint8')
load_bw_f32 = ImgLoader(ndim=2, dtype='float32')
load_bw_u8 = ImgLoader(ndim=2, dtype='uint8')

