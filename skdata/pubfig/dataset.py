"""
http://www.cs.columbia.edu/CAVE/databases/pubfig/

The PubFig database is a large, real-world face dataset consisting of 58,797
images of 200 people collected from the internet. Unlike most other existing
face datasets, these images are taken in completely uncontrolled situations
with non-cooperative subjects. Thus, there is large variation in pose,
lighting, expression, scene, camera, imaging conditions and parameters, etc.

Citation

The database is made available only for non-commercial use. If you use this
dataset, please cite the following paper:

    "Attribute and Simile Classifiers for Face Verification,"
    Neeraj Kumar, Alexander C. Berg, Peter N. Belhumeur, and Shree K. Nayar,
    International Conference on Computer Vision (ICCV), 2009.

"""

# Copyright (C) 2013
# Authors: James Bergstra <james.bergstra@uwaterloo.ca>

# License: Simplified BSD
import os

from skdata.data_home import get_data_home
from skdata.utils import download


def url_of(filename):
    root = 'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/'
    return root + filename

urls = dict([(filename, url_of(filename)) for filename in [
    'dev_people.txt',
    'dev_urls.txt',
    'eval_people.txt',
    'eval_urls.txt',
    'pubfig_labels.txt',
    'pubfig_full.txt',
    'pubfig_attributes.txt',
        ]])

md5s = {
    'dev_people.txt': None,
    'dev_urls.txt': None,
    'eval_people.txt': None,
    'eval_urls.txt': None,
    'pubfig_labels.txt': None,
    'pubfig_full.txt': None,
    'pubfig_attributes.txt': None,
        }

class PubFig(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.name, *suffix_paths)

    def fetch(self, download_if_missing=True):
        """Download and extract the dataset."""

        home = self.home()

        if not os.path.exists(home):
            if download_if_missing:
                raise NotImplementedError()
            else:
                raise IOError("'%s' does not exists!" % home)

        for filename, url in urls.items():
            download(url, self.home(filename), md5=md5s[filename])

        return  # XXX REST IS CUT AND PASTE FROM ELSEWHERE

        for fkey, (fname, sha1) in self.FILES.iteritems():
            url = path.join(BASE_URL, fname)
            basename = path.basename(url)
            archive_filename = path.join(home, basename)
            if not path.exists(archive_filename):
                if not download_if_missing:
                    return
                if not path.exists(home):
                    os.makedirs(home)
                download(url, archive_filename, sha1=sha1)

