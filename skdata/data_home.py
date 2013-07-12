"""Manage the scikit-data cache directory.

This folder is used by some large dataset loaders to avoid downloading the data
several times.

By default the data dir is set to a folder named '.scikit-data'
in the user home folder.  This directory can be specified prior to importing
this module via the SKDATA_ROOT environment variable.

After importing the module that environment variable is no longer consulted,
and a module-level variable called DATA_HOME becomes the new point of reference.

DATA_HOME can be read or modified directly, or via the two functions
get_data_home() and set_data_home(). Compared to the raw DATA_HOME variable,
the functions have the side effect of ensuring that the DATA_HOME directory
exists, and is readable.

"""

import os
import shutil

DATA_HOME = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            'SKDATA_ROOT',
            os.path.join('~', '.skdata'))))

def get_data_home():
    if not os.path.isdir(DATA_HOME):
        os.makedirs(DATA_HOME)
    # XXX: ensure it is dir and readable
    return DATA_HOME


def set_data_home(newpath):
    global DATA_HOME
    DATA_HOME = newpath
    return get_data_home()


def clear_data_home():
    """Delete all the content of the data home cache."""
    data_home = get_data_home()
    shutil.rmtree(data_home)
