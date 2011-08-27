"""
Utilities to access data cache directories.

"""

import os
import sys

# -- root directory for Dataset and DatasetBlob objects to store data.
SCIKITS_DATA = os.path.expanduser(
        os.environ.get('SCIKITS_DATA',
            os.path.join(os.environ.get('HOME'), '.scikits.data')))

def get_cache_dir(dataset_name):
    """
    Return readable subdirectory path: `SCIKITS_DATA/dataset_name`.

    This method tries to make the directory if it is not readable.
    """
    rval = os.path.join(SCIKITS_DATA, dataset_name)
    try:
        os.listdir(rval)  # succeeds iff rval is already readable
    except OSError:
        os.makedirs(rval)   # succeeds iff rval is now readable
    return rval
