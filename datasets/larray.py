"""
LazyArray
"""
import sys
import StringIO
import numpy as np
import logging
logger = logging.getLogger(__name__)

class InferenceError(Exception):
    """Information about a lazily-evaluated quantity could not be inferred"""

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
        return pprint_str(self)

    def __print__(self):
        return self.__repr__()

    def clone(self, given):
        raise NotImplementedError('override-me')

    def inputs(self):
        raise NotImplementedError('override-me')

    def lazy_inputs(self):
        return [ii for ii in self.inputs() if is_larray(ii)]


class larray(lazy):
    def loop(self):
        return loop(self)


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

    def __get_shape(self):
        shape_0 = len(self)
        shape_rest = self.fn.rval_getattr('shape', objs=self.objs)
        return (shape_0,) + shape_rest
    shape = property(__get_shape)

    def __get_dtype(self):
        return self.fn.rval_getattr('dtype', objs=self.objs)
    dtype = property(__get_dtype)

    def __get_ndim(self):
        return 1 + self.fn.rval_getattr('ndim', objs=self.objs)
    ndim = property(__get_ndim)

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
        return numpy.asarray([self.fn(*[o[i] for o in self.objs])
                for i in xrange(len(self))])

    def __print__(self):
        return 'lmap(%s, ...)' % (self.fn.__name__,)

    def clone(self, given):
        return lmap(self.fn, *[given_get(given, obj) for obj in self.objs])

    def inputs(self):
        return list(self.objs)

def lzip(*arrays):
    # XXX: make a version of this method that supports call_batch
    class fn(object):
        def __call__(self, *args):
            return numpy.asarray(args)
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


def pprint(thing, prefix='', buf=None):
    if buf is None:
        buf = sys.stdout
    if hasattr(thing, '__print__'):
        print >> buf, '%s%s'%(prefix, thing.__print__())
    else:
        print >> buf, '%s%s'%(prefix, str(thing))
    if is_larray(thing):
        for ii in thing.inputs():
            pprint(ii, prefix+'    ', buf)


def pprint_str(thing):
    sio = StringIO.StringIO()
    pprint(thing, '', sio)
    return sio.getvalue()

#
# Stuff to consider factoring out somewhere because of the imread dependency
#

def Flatten(object):
    def rval_getattr(self, attr, objs):
        if attr == 'shape':
            shp = objs[0].shape[1:]
            if None in shp:
                return (None,)
            else:
                return (numpy.prod(shp),)
        if attr == 'ndim':
            return 1
        if attr == 'dtype':
            return objs[0].dtype
        raise AttributeError(attr)
    def __call__(self, thing):
        return numpy.flatten(thing)

def flatten_elements(seq):
    return lmap(Flatten(), seq)


