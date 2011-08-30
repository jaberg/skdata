"""
LazyArray
"""
import sys
import StringIO
import numpy as np

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
    print arrays
    return lmap((lambda *args: args), *arrays)


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


if 0:
    class AdvIndexable1D(object):
        """
        Suitable base for finite array.
        """
        def __getitem__(self, idx):
            if isinstance(idx, int):  # does this catch e.g. np.uint8?
                return self.get_int(idx)
            elif isinstance(idx, slice):
                return self.get_slice(idx)
            else:
                return self.get_array(idx)

        def get_int(self, int_idx):
            raise NotImplementedError('override-me')

        def get_slice(self, slice_idx):
            start, stop, step = slice_idx.indices(len(self))
            return SlicedAdvIdx1D(self, start, stop, step)

        def get_array(self, array_idx):
            rval = [self.get_int(i) for i in array_idx]
            return rval

        def __array__(self):
            return np.array(self.src[range(len(self))])


    class SlicedAdvIdx1D(AdvIndexable1D):
        def __init__(self, src, start, stop, step):
            self.src = src
            self.start = start
            self.stop = stop
            self.step = step

        def get_int(self, int_idx):
            if int_idx >= 0:
                idx = self.start + self.step * int_idx
                if idx < self.stop:
                    return self.src[idx]
                else:
                    raise IndexError()
            else:
                idx = self.stop - 1 + self.step * int_idx
                if idx >= 0:
                    return self.src[idx]
                else:
                    raise IndexError()

    class Map(AdvIndexable1D):
        def __init__(self, src, f, f_batch=None):
            self.src = src
            self.f = f
            self.f_batch = f_batch

        def get_int(self, int_idx):
            return self.f_int(self.src[int_idx])

        def get_slice(self, slice_idx):
            tmp = self.__getitem__(slice_idx)
            if self.f_batch is None:
                rval = [self.f(t) for t in tmp]
            else:
                rval = self.f_batch(tmp)
            return numpy.asarray(rval)
        
        def get_array(self, array_idx):
            tmp = self.src[array_idx]
            if self.f_batch is None:
                rval = [self.f(t) for t in tmp]
            else:
                rval = self.f_batch(tmp)
            return numpy.asarray(rval)


