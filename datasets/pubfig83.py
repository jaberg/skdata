"""
Zak - What exactly is this dataset? It's public and published now right?
"""
import os
import sys

from .base import get_data_home, Bunch
from .utils import memoize

genders = ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'F', 'F',
'F', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M',
'M', 'F', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F',
'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M',
'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'M',
'F', 'F', 'M', 'M', 'M']


def pubfig83_join(*names):
    return os.path.join(get_data_home(), 'pubfig83', *names)


def fetch():
    """Download and extract the dataset."""
    dataset_dir = pubfig83_join()
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    # XXX: use urllib instead of wget
    os.system('wget http://www.eecs.harvard.edu/~zak/pubfig83/pubfig83_first_draft.tgz -P ' + dataset_dir)
    # XXX: use python's tar wrapper instead of this command which
    #      will (I think) not work on windows
    os.system('cd ' + dataset_dir + '; tar -xzf pubfig83_first_draft.tgz')

@memoize
def load():
    """
    Return a list of dictionaries. Each dict contains
        gender -> 'M' or 'F'
        name -> str
        id -> int
        jpgfile -> relpath
    """
    names = os.listdir(pubfig83_join('pubfig83'))
    names.sort()
    assert len(names) == len(genders)
    genders_names_pics = []
    ind = 0
    for gender, name in zip(genders, names):
        pics = os.listdir(pubfig83_join('pubfig83', name))
        pics.sort()
        for pic in pics:
            genders_names_pics.append(dict(
                gender=gender,
                name=name,
                id=ind,
                jpgfile=pic))
            ind +=1
    return genders_names_pics

class PubFig83(Bunch):
    def __init__(self, download_if_missing=False):
        dataset_path = pubfig83_join()
        if not os.path.exists(dataset_path):
            if download_if_missing:
                self.fetch()
            else:
                raise IOError(dataset_path)
        genders_names_pics = load()
        Bunch.__init__(self,
                meta=genders_names_pics,
                gender = [m['gender'] for m in genders_names_pics],
                name =[m['name'] for m in genders_names_pics],
                jpgfile = [m['jpgfile'] for m in genders_names_pics],
                img_fullpath = [pubfig83_join('pubfig83', m['name'], m['jpgfile'])
                    for m in genders_names_pics])

#
# Drivers for scikits.data/bin executables
#

def main_fetch():
    """compatibility with bin/datasets-fetch"""
    fetch()


def main_show():
    """compatibility with bin/datasets-show"""
    from glviewer import glumpy_viewer, command, glumpy
    import larray
    bunch = PubFig83()
    glumpy_viewer(
            img_array=larray.img_load(bunch.img_fullpath),
            arrays_to_print=[bunch.name])
