"""
XXX Description of dataset (Zak?)

XXX Citation

"""
import os
import sys

from data_home import get_data_home
import utils, utils.image

_genders = ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'F',
'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F',
'M', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F',
'F', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F',
'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'M',
'M', 'F', 'F', 'M', 'M', 'M']

class PubFig83(object):
    """
    XXX Technical description of class (attributes, methods, etc.) rather than
    descr of dataset (which is above in the file).

    self.meta a list of dictionaries. Each one contains
        gender: 'M' or 'F'
        name: str
        id: int
        jpgfile: relpath
    """

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta
    #
    # Standard dataset object interface
    #

    def __get_meta(self):
        try:
            return self._meta
        except AttributeError:
            self.fetch(download_if_missing=True)
            self._meta = self.build_meta()
            return self._meta
    meta = property(__get_meta)

    #
    # Helper routines
    #

    def home(self, *names):
        return os.path.join(get_data_home(), 'pubfig83', *names)

    def build_meta(self):
        names = sorted(os.listdir(self.home('pubfig83')))
        assert len(names) == len(_genders)
        meta = []
        ind = 0
        for gender, name in zip(_genders, names):
            for pic in sorted(os.listdir(self.home('pubfig83', name))):
                meta.append(dict(
                    gender=gender,
                    name=name,
                    id=ind,
                    jpgfile=pic))
                ind +=1
        return meta

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""
        dataset_dir = self.home()
        if os.path.exists(dataset_dir):
            return
        if not download_if_missing:
            raise IOError(dataset_dir)
        os.makedirs(dataset_dir)
        # XXX: use urllib instead of wget
        os.system('wget http://www.eecs.harvard.edu/~zak/pubfig83/pubfig83_first_draft.tgz -P ' + dataset_dir)
        # XXX: use python's tar wrapper instead of this command which
        #      will (I think) not work on windows
        os.system('cd ' + dataset_dir + '; tar -xzf pubfig83_first_draft.tgz')

    def erase(self):
        if isdir(self.home()):
            shutil.rmtree(self.home())

    def image_path(self, m):
        return self.home('pubfig83', m['name'], m['jpgfile'])

    #
    # Standard Tasks
    # --------------
    #

    def raw_recognition_task(self):
        names = [m['name'] for m in self.meta]
        paths = [self.image_path(m) for m in self.meta]
        labels = utils.int_labels(names)
        return paths, labels

    def raw_gender_task(self):
        genders = [m['gender'] for m in self.meta]
        paths = [self.image_path(m) for m in self.meta]
        return paths, utils.int_labels(genders)


#
# Drivers for scikits.data/bin executables
#

def main_fetch():
    """compatibility with bin/datasets-fetch"""
    fetch()

def main_show():
    """compatibility with bin/datasets-show"""
    from utils.glviewer import glumpy_viewer, command, glumpy
    import larray
    pf = PubFig83()
    names = [m['name'] for m in pf.meta]
    paths = [pf.image_path(m) for m in pf.meta]
    glumpy_viewer(
            img_array=larray.lmap(utils.image.ImgLoader(), paths),
            arrays_to_print=[names])


