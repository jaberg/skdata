from os import path

__all__ = ['get_my_path',
           'get_my_path_basename',
          ]


def get_my_path(my_file=None):
    if my_file is None:
        import inspect
        caller = inspect.currentframe().f_back
        my_file = caller.f_globals['__file__']
    return path.dirname(path.abspath(my_file))


def get_my_path_basename(my_file=None):
    if my_file is None:
        import inspect
        caller = inspect.currentframe().f_back
        my_file = caller.f_globals['__file__']
    dirname = path.dirname(path.abspath(my_file))
    basename = path.basename(dirname)
    return basename
