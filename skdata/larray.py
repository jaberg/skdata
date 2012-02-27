"""
LazyArray
"""
import cPickle
import logging
import os
import StringIO
import subprocess
import sys
import numpy as np

from .data_home import get_data_home

logger = logging.getLogger(__name__)

class InferenceError(Exception):
    """Information about a lazily-evaluated quantity could not be inferred"""

class UnknownShape(InferenceError):
    """Shape could not be inferred"""

def is_int_idx(idx):
    #XXX: add numpy int types
    return isinstance(idx,
            (int,))


def is_larray(thing):
    return isinstance(thing, larray)


def given_get(given, thing):
    try:
        return given.get(thing, thing)
    except TypeError:
        return thing


class lazy(object):
    def __str__(self):
        return lprint_str(self)

    def __print__(self):
        return self.__repr__()

    def clone(self, given):
        """
        Return a new object that will behave like self, but with new
        inputs.  For any input `obj` to self, the clone should have
        input `given_get(given, self.obj)`.
        """
        raise NotImplementedError('override-me',
                                  (self.__class__, 'clone'))

    def inputs(self):
        raise NotImplementedError('override-me',
                                 (self.__class__, 'inputs'))

    def lazy_inputs(self):
        return [ii for ii in self.inputs() if is_larray(ii)]


class larray(lazy):
    """
    A class inheriting from larray is like `lazy` but promises
    additionally to provide three attributes or properties:
    - .shape
    - .ndim
    - .dtype
    - .strides

    These should be used to maintain consistency with numpy.
    """
    def loop(self):
        return loop(self)

    def __len__(self):
        return self.shape[0]


class lmap(larray):
    """
    Return a lazily-evaluated mapping.

    fn can be a normal lambda expression, but it can also respond to the
    following attributes:

    - call_batch

    - rval_getattr
        `fn.rval_getattr(name)` if it returns, must return the same thing as
        `getattr(fn(*args), name)` would return.
    """
    #TODO: add kwarg to specify f_map implementation
    #      that is drop-in for map(f, *args)
    def __init__(self, fn, obj0, *objs, **kwargs):
        ragged = kwargs.pop('ragged', False)
        f_map = kwargs.pop('f_map', None)
        if kwargs:
            raise TypeError('unrecognized kwarg', kwargs.keys())

        self.fn = fn
        self.objs = [obj0] + list(objs)
        self.ragged = ragged
        self.f_map = f_map
        if not ragged:
            for o in objs:
                if len(obj0) != len(o):
                    raise ValueError('objects have different length')

    def __len__(self):
        if len(self.objs)>1:
            return min(*[len(o) for o in self.objs])
        else:
            return len(self.objs[0])

    @property
    def shape(self):
        shape_0 = len(self)
        if hasattr(self.fn, 'rval_getattr'):
            shape_rest = self.fn.rval_getattr('shape', objs=self.objs)
            return (shape_0,) + shape_rest
        raise UnknownShape() 

    @property
    def dtype(self):
        return self.fn.rval_getattr('dtype', objs=self.objs)

    @property
    def ndim(self):
        return 1 + self.fn.rval_getattr('ndim', objs=self.objs)

    def __getitem__(self, idx):
        if is_int_idx(idx):
            return self.fn(*[o[idx] for o in self.objs])
        else:
            try:
                tmps = [o[idx] for o in self.objs]
            except TypeError:
                # advanced indexing failed, try one element at a time
                return [self.fn(*[o[i] for o in self.objs])
                        for i in idx]

            # we loaded our args by advanced indexing
            if self.f_map:
                return self.f_map(*tmps)
            else:
                return map(self.fn, *tmps)

    def __array__(self):
        #XXX: use self.batch_len to produce this more efficiently
        return np.asarray([self.fn(*[o[i] for o in self.objs])
                for i in xrange(len(self))])

    def __print__(self):
        return 'lmap(%s, ...)' % (self.fn.__name__,)

    def clone(self, given):
        return lmap(self.fn, *[given_get(given, obj) for obj in self.objs])

    def inputs(self):
        return list(self.objs)


class RvalGetattr(object):
    """
    See `lmap_info`
    """
    def __init__(self, info):
        self.info = info

    def __call__(self, name, objs=None):
        try:
            return self.info[name]
        except KeyError:
            raise InferenceError(name)


def lmap_info(**kwargs):
    """Decorator for providing information for lmap

    >>> @lmap_info(shape=(10, 20), dtype='float32')
    >>> def foo(i):
    >>>     return np.zeros((10, 20), dtype='float32') + i
    >>>
    """

    # -- a little hack of convenience
    if 'shape' in kwargs:
        if 'ndim' in kwargs:
            assert len(kwargs['shape']) == kwargs['ndim']
        else:
            kwargs['ndim'] = len(kwargs['shape'])

    def wrapper(f):
        f.rval_getattr = RvalGetattr(kwargs)
        return f

    return wrapper


def lzip(*arrays):
    # XXX: make a version of this method that supports call_batch
    class fn(object):
        __name__ = 'lzip'
        def __call__(self, *args):
            return np.asarray(args)
        def rval_getattr(self, name, objs=None):
            if name == 'shape':
                shps = [o.shape for o in objs]
                shp1 = len(objs)
                # if all the rest of the shapes are equal
                # then we have something to say,
                # otherwise no idea.
                if all(shps[0][1:] == s[1:] for s in shps):
                    return (shp1,) + shps[0][1:]
                else:
                    raise InferenceError('dont know shape')
                raise NotImplementedError()
            if name == 'dtype':
                # if a shape cannot be inferred, then the
                # zip result might be ragged, in which case the dtype would be
                # `object`.
                shape = self.rval_getattr('shape', objs)
                # postcondition: result is ndarray-like

                if all(o.dtype == objs[0].dtype for o in objs[1:]):
                    return objs[0].dtype
                else:
                    # XXX upcasting rules
                    raise NotImplementedError()
            if name == 'ndim':
                # if a shape cannot be inferred, then the
                # zip result might be ragged, in which case the dtype would be
                # `object`.
                shape = self.rval_getattr('shape', objs)
                return len(shape)
            raise AttributeError(name)
    return lmap(fn(), *arrays)


class loop(larray):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, idx):
        if is_int_idx(idx):
            return self.obj[idx % len(self.obj)]
        elif isinstance(idx, slice):
            raise NotImplementedError()
        elif isinstance(idx, (tuple, list, np.ndarray)):
            idx = np.asarray(idx) % len(self.obj)
            #XXX: fallback if o does not support advanced indexing
            return self.obj[idx]

    def clone(self, given):
        return loop(given_get(given, self.obj))

    def inputs(self):
        return [self.obj]


class reindex(larray):
    def __init__(self, obj, imap):
        self.obj = obj
        self.imap = np.asarray(imap)
        if 'int' not in str(self.imap.dtype):
            #XXX: diagnostic info
            raise TypeError()

    def __len__(self):
        return len(self.imap)

    def __getitem__(self, idx):
        mapped_idx = self.imap[idx]
        try:
            return self.obj[mapped_idx]
        except TypeError:
            # XXX: try this, and restore original exception on failure
            return [self.obj[ii] for ii in mapped_idx]

    def __get_shape(self):
        return (len(self),) + self.obj.shape[1:]
    shape = property(__get_shape)

    def __get_dtype(self):
        return self.obj.dtype
    dtype = property(__get_dtype)

    def __get_ndim(self):
        return self.obj.ndim
    ndim = property(__get_ndim)

    def clone(self, given):
        return reindex(
                given_get(given, self.obj),
                given_get(given, self.imap)
                )

    def inputs(self):
        return [self.obj, self.imap]


def clone_helper(thing, given):
    if thing in given:
        return
    for ii in thing.lazy_inputs():
        clone_helper(ii, given)
    given[thing] = thing.clone(given)


def clone(thing, given):
    _given = dict(given)
    clone_helper(thing, _given)
    return _given[thing]


def lprint(thing, prefix='', buf=None):
    if buf is None:
        buf = sys.stdout
    if hasattr(thing, '__print__'):
        print >> buf, '%s%s'%(prefix, thing.__print__())
    else:
        print >> buf, '%s%s'%(prefix, str(thing))
    if is_larray(thing):
        for ii in thing.inputs():
            lprint(ii, prefix+'    ', buf)


def lprint_str(thing):
    sio = StringIO.StringIO()
    lprint(thing, '', sio)
    return sio.getvalue()


def Flatten(object):
    def rval_getattr(self, attr, objs):
        if attr == 'shape':
            shp = objs[0].shape[1:]
            if None in shp:
                return (None,)
            else:
                return (np.prod(shp),)
        if attr == 'ndim':
            return 1
        if attr == 'dtype':
            return objs[0].dtype
        raise AttributeError(attr)
    def __call__(self, thing):
        return np.flatten(thing)

def flatten_elements(seq):
    return lmap(Flatten(), seq)


memmap_README = """\
Memmap files created by LazyCacheMemmap

  data.raw - memmapped array data file, no header
  valid.raw - memmapped array validity file, no header
  header.pkl - python pickle of meta-data (dtype, shape) for data.raw

The validitiy file is a byte array that indicates which elements of
data.raw are valid.  If valid.raw byte `i` is 1, then the `i`'th tensor
slice of data.raw has been computed and is usable. If it is 0, then it
has not been computed and the slice value is undefined. No other values
should appear in the valid.raw array.
"""

class CacheMixin(object):
    def populate(self, batchsize=1):
        """
        Populate a lazy array cache node by iterating over the source in
        increments of `batchsize`.
        """
        if batchsize <= 0:
            raise ValueError('non-positive batch size')
        if batchsize == 1:
            for i in xrange(self.shape[0]):
                self[i]
        else:
            while i < self.shape[0]:
                self[i:i + batchsize]
                i += batchsize

    @property
    def shape(self):
        return self.obj.shape

    @property
    def dtype(self):
        return self.obj.dtype

    @property
    def ndim(self):
        return self.obj.ndim

    def inputs(self):
        return [self.obj]

    def __getitem__(self, item):
        if isinstance(item, (int, np.int)):
            if self._valid[item]:
                return self._data[item]
            else:
                obj_item = self.obj[item]
                self._data[item] = obj_item
                self._valid[item] = 1
                self.rows_computed += 1
                return self._data[item]
        else:
            # could be a slice, an intlist, a tuple
            v = self._valid[item]
            assert v.ndim == 1
            if np.all(v):
                return self._data[item]
            # otherwise at least some of the requested elements must be
            # computed
            # XXX: this brute-force recomputes all of the requested elements
            self.rows_computed += v.sum()
            values = self.obj[item]
            self._valid[item] = 1
            self._data[item] = values
            return self._data[item]


class cache_memory(CacheMixin, larray):
    """
    Provide a lazily-filled cache of a larray (obj) via an in-memmory
    array.
    """

    def __init__(self, obj):
        """
        If new files are created, then `msg` will be written to README.msg
        """
        self.obj = obj
        self._data = np.empty(obj.shape, dtype=obj.dtype)
        self._valid = np.zeros(len(obj), dtype='int8')
        self.rows_computed = 0

    def clone(self, given):
        return self.__class__(obj=given_get(given, self.obj))

    
class cache_memmap(CacheMixin, larray):
    """
    Provide a lazily-filled cache of a larray (obj) via a memmap file
    associated with (name).


    The memmap will be stored in `basedir`/`name` which defaults to
    `cache_memmap.ROOT`/`name`,
    which defaults to '~/.skdata/memmaps'/`name`.
    """

    ROOT = os.path.join(get_data_home(), 'memmaps')

    def __init__(self, obj, name, basedir=None, msg=None):
        """
        If new files are created, then `msg` will be written to README.msg
        """

        self.obj = obj
        if basedir is None:
            basedir = self.ROOT
        self.dirname = dirname = os.path.join(basedir, name)
        subprocess.call(['mkdir', '-p', dirname])

        data_path = os.path.join(dirname, 'data.raw')
        valid_path = os.path.join(dirname, 'valid.raw')
        header_path = os.path.join(dirname, 'header.pkl')

        try:
            dtype, shape = cPickle.load(open(header_path))
            if obj is None or (dtype == obj.dtype and shape == obj.shape):
                mode = 'r+'
            else:
                mode = 'w+'
        except IOError:
            dtype = obj.dtype
            shape = obj.shape
            mode = 'w+'

        logger.info('Creating memmap %s for features of shape %s' % (
                data_path,
                str(shape)))

        self._data = np.memmap(data_path,
            dtype=dtype,
            mode=mode,
            shape=shape)

        self._valid = np.memmap(valid_path,
            dtype='int8',
            mode=mode,
            shape=(shape[0],))

        if mode == 'w+':
            # initialize a new set of files
            cPickle.dump((dtype, shape),
                         open(header_path, 'w'))
            # mark all memmap elements as uncomputed
            self._valid[:] = 0

            open(os.path.join(dirname, 'README.txt'), 'w').write(
                memmap_README)
            if msg is not None:
                open(os.path.join(dirname, 'README.msg'), 'w').write(
                    str(msg))
            warning = ( 'WARNING_THIS_DIR_WILL_BE_DELETED'
                        '_BY_cache_memmap.delete_files()')
            open(os.path.join(dirname, warning), 'w').close()

        self.rows_computed = 0

    def delete_files(self):
        subprocess.call(['rm', '-Rf', self.dirname])

    def clone(self, given):
        raise NotImplementedError()
        return self.__class__(
            obj=given_get(given, self.obj),
            dirname=self.dirname + '_clone')
    # XXX: What if you clone more than once? This implementation would
    #      cause interference
