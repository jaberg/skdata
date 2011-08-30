"""Loader for the Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/

Each picture is centered on a single face. The typical task is called
Face Verification: given a pair of two pictures, a binary classifier
must predict whether the two images are from the same person.

An alternative task, Face Recognition or Face Identification is:
given the picture of the face of an unknown person, identify the name
of the person by refering to a gallery of previously seen pictures of
identified persons.

Both Face Verification and Face Recognition are tasks that are typically
performed on the output of a model trained to perform Face Detection. The
most popular model for Face Detection is called Viola-Johns and is
implemented in the OpenCV library. The LFW faces were extracted by this face
detector from various online websites.
"""
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

from os import listdir, makedirs, remove
from os.path import join, exists, isdir
import sys

import logging
import numpy as np
import urllib

from .base import get_data_home, Bunch
import larray

logger = logging.getLogger(__name__)

try:
    try:
        from scipy.misc import imread
    except ImportError:
        from scipy.misc.pilutil import imread
    from scipy.misc import imresize
except ImportError:
    logger.warn("The Python Imaging Library (PIL)"
            " is required to load data from jpeg files.")


BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
ARCHIVE_NAME = "lfw.tgz"
FUNNELED_ARCHIVE_NAME = "lfw-funneled.tgz"
TARGET_FILENAMES = [
    'pairsDevTrain.txt',
    'pairsDevTest.txt',
    'pairs.txt',
]


def scale_face(face):
    """Scale back to 0-1 range in case of normalization for plotting"""
    scaled = face - face.min()
    scaled /= scaled.max()
    return scaled


#
# Common private utilities for data fetching from the original LFW website
# local disk caching, and image decoding.
#

def memoize(f):
    """Simple decorator hashes *args to f, stores rvals in dict.

    This simple decorator works in memory only, does not persist between
    processes.
    """
    cache = {}
    def cache_f(*args):
        if args in cache:
            return cache[args]
        rval = f(*args)
        cache[args] = rval
        return rval
    return cache_f


def check_fetch_lfw(data_home=None, funneled=True, download_if_missing=True):
    """Helper function to download any missing LFW data.

    Call this function with no arguments to download the funneled LFW dataset to the standard
    location. This downloads about 200MB.
    """
    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, "lfw_home")

    if funneled:
        archive_path = join(lfw_home, FUNNELED_ARCHIVE_NAME)
        data_folder_path = join(lfw_home, "lfw_funneled")
        archive_url = BASE_URL + FUNNELED_ARCHIVE_NAME
    else:
        archive_path = join(lfw_home, ARCHIVE_NAME)
        data_folder_path = join(lfw_home, "lfw")
        archive_url = BASE_URL + ARCHIVE_NAME

    if not exists(lfw_home):
        makedirs(lfw_home)

    for target_filename in TARGET_FILENAMES:
        target_filepath = join(lfw_home, target_filename)
        if not exists(target_filepath):
            if download_if_missing:
                url = BASE_URL + target_filename
                logger.warn("Downloading LFW metadata: %s => %s" % (
                    url, target_filepath))
                downloader = urllib.urlopen(BASE_URL + target_filename)
                data = downloader.read()
                open(target_filepath, 'wb').write(data)
            else:
                raise IOError("%s is missing" % target_filepath)

    if not exists(data_folder_path):

        if not exists(archive_path):
            if download_if_missing:
                logger.warn("Downloading LFW data (~200MB): %s => %s" %(
                        archive_url, archive_path))
                downloader = urllib.urlopen(archive_url)
                data = downloader.read()
                # don't open file until download is complete
                open(archive_path, 'wb').write(data)
            else:
                raise IOError("%s is missing" % target_filepath)

        import tarfile
        logger.info("Decompressing the data archive to %s", data_folder_path)
        tarfile.open(archive_path, "r:gz").extractall(path=lfw_home)
        remove(archive_path)

    return lfw_home, data_folder_path


@memoize
def names_paths(data_folder_path, min_faces_per_person):
    """Return two corresponding lists of names and image paths.

    This routine walks the data_folder_path to build these lists.
    """
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in listdir(folder_path)]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)
    return person_names, file_paths


class img_loader(object):
    """This class is an image-loading filter for use with larray.map"""

    #TODO: Consider factoring this out of lfw - it works for general image files

    def __init__(self, slice_, color, resize):
        self.color = color
        self.resize = resize

        # compute the portion of the images to load to respect the slice_ parameter
        # given by the caller
        default_slice = (slice(0, 250), slice(0, 250))
        if slice_ is None:
            self.slice_ = default_slice
        else:
            self.slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))
        del slice_

        h_slice, w_slice = self.slice_
        self.h = (h_slice.stop - h_slice.start) / (h_slice.step or 1)
        self.w = (w_slice.stop - w_slice.start) / (w_slice.step or 1)

        if self.resize is not None:
            self.resize = float(self.resize)
            self.h = int(self.resize * self.h)
            self.w = int(self.resize * self.w)

    def __call__(self, file_path):
        return self.call_batch([file_path])[0]

    def rval_getattr(self, attr, objs):
        if attr == 'shape':
            if self.color:
                return (self.h, self.w, 3)
            else:
                return (self.h, self.w)
        if attr == 'dtype':
            return np.float32
        if attr == 'ndim':
            return 3 if self.color else 2
        raise AttributeError(attr)

    def call_batch(self, file_paths):
        # allocate some contiguous memory to host the decoded image slices
        n_faces = len(file_paths)
        if not self.color:
            faces = np.zeros((n_faces, self.h, self.w), dtype=np.float32)
        else:
            faces = np.zeros((n_faces, self.h, self.w, 3), dtype=np.float32)

        # iterate over the collected file path to load the jpeg files as numpy
        # arrays
        for i, file_path in enumerate(file_paths):
            if i % 1000 == 0:
                logger.info("Loading face #%05d / %05d", i + 1, n_faces)
            face = np.asarray(imread(file_path)[self.slice_], dtype=np.float32)
            face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
            if self.resize is not None and self.resize != 1.0:
                face = imresize(face, self.resize)
            if not self.color:
                # average the color channels to compute a gray levels
                # representaion
                # XXX: there are some standard constants for doing this
                #      that weight channels differently
                face = face.mean(axis=2)
            faces[i, ...] = face
        return faces


#
# Task #1:  Face Identification on picture with names
#

#XXX take the post-processing out of the dataset API.
#    Implement resizing, greyscaling, cropping
#    as generic post-processing steps.

def load_lfw_people(data_home=None, funneled=True, resize=0.5,
                    min_faces_per_person=None, color=False,
                    slice_=(slice(70, 195), slice(78, 172)),
                    download_if_missing=False):
    """Loader for the Labeled Faces in the Wild (LFW) people dataset

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on the
    official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face. Each pixel of each channel
    (color in RGB) is encoded by a float in range 0.0 - 1.0.

    The task is called Face Recognition (or Identification): given the
    picture of a face, find the name of the person given a training set
    (gallery).

    Parameters
    ----------
    data_home: optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    funneled: boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize: float, optional, default 0.5
        Ratio used to resize the each face picture.

    min_faces_per_person: int, optional, default None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color: boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than than the shape with color = False.

    slice_: optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.
    """
    lfw_home, data_folder_path = check_fetch_lfw(
        data_home=data_home, funneled=funneled,
        download_if_missing=download_if_missing)
    logger.info('Loading LFW people faces from %s', lfw_home)
    # postcondition: data is downloaded to lfw_home

    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = names_paths(data_folder_path, min_faces_per_person)
    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    faces = larray.lmap(img_loader(slice_, color, resize), file_paths)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces = larray.reindex(faces, indices)
    target = target[indices]

    # pack the results as a Bunch instance
    return Bunch(
            imgs=faces,
            target=target,
            target_names=target_names,
            names=larray.reindex(target_names, target),
            DESCR="LFW faces dataset")


#
# Task #2:  Face Verification on pairs of face pictures
#

@memoize
def img_pairs(index_file_path, data_folder_path):
    """
    index_file is a text file whose rows are one of
        name1 I name2 J   # non-matching pair    (name1, I), (name2, J)
        name I J          # matching image pair  (name, I), (name, J)
        N M               # not sure about these ones

    This function returns three lists:
    - left image paths
    - right image paths
    - binary ndarray: targets match?
    """
    splitted_lines = [l.strip().split('\t')
                      for l in open(index_file_path, 'rb').readlines()]
    pair_specs = [sl for sl in splitted_lines if len(sl) > 2]
    n_pairs = len(pair_specs)

    # interating over the metadata lines for each pair to find the filename to
    # decode and load in memory
    target = np.zeros(n_pairs, dtype=np.int)
    left_paths = []
    right_paths = []
    person_names, file_paths = names_paths(data_folder_path, 0)

    for i, components in enumerate(pair_specs):
        if len(components) == 3:
            target[i] = 1
            left = (components[0].replace('_', ' '), int(components[1]) - 1)
            right = (components[0].replace('_', ' '), int(components[2]) - 1)
        elif len(components) == 4:
            target[i] = 0
            left = (components[0].replace('_', ' '), int(components[1]) - 1)
            right = (components[2].replace('_', ' '), int(components[3]) - 1)
        else:
            raise ValueError("invalid line %d: %r" % (i + 1, components))

        # a dictionary would make this more readable.
        left_paths.append(file_paths[person_names.index(left[0]) + left[1]])
        right_paths.append(file_paths[person_names.index(right[0]) + right[1]])

    return left_paths, right_paths, target


def load_lfw_pairs(subset='train', data_home=None, funneled=True, resize=0.5,
                   color=False, slice_=(slice(70, 195), slice(78, 172)),
                   download_if_missing=False):
    """Loader for the Labeled Faces in the Wild (LFW) pairs dataset

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on the
    official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face. Each pixel of each channel
    (color in RGB) is encoded by a float in range 0.0 - 1.0.

    The task is called Face Verification: given a pair of two pictures,
    a binary classifier must predict whether the two images are from
    the same person.

    In the official `README.txt`_ this task is described as the
    "Restricted" task.  The "Unrestricted" variant is not currently supported.

      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

    Parameters
    ----------
    subset: optional, default: 'train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home: optional, default: None
        Specify another download and cache folder for the datasets. By
        default all scikit learn data is stored in '~/scikit_learn_data'
        subfolders.

    funneled: boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize: float, optional, default 0.5
        Ratio used to resize the each face picture.

    color: boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than than the shape with color = False.

    slice_: optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.
    """
    lfw_home, data_folder_path = check_fetch_lfw(
        data_home=data_home, funneled=funneled,
        download_if_missing=download_if_missing)
    logger.info('Loading %s LFW pairs from %s', subset, lfw_home)
    # postcondition: data is downloaded to lfw_home

    # select the right metadata file according to the requested subset
    label_filenames = {
        'train': 'pairsDevTrain.txt',
        'test': 'pairsDevTest.txt',
        '10_folds': 'pairs.txt',
    }
    if subset not in label_filenames:
        raise ValueError("subset='%s' is invalid: should be one of %r" % (
            subset, list(sorted(label_filenames.keys()))))
    index_file_path = join(lfw_home, label_filenames[subset])

    lpaths, rpaths, target = img_pairs(index_file_path, data_folder_path)
    left_imgs = larray.lmap(img_loader(slice_, color, resize), lpaths)
    right_imgs = larray.lmap(img_loader(slice_, color, resize), rpaths)
    pairs = larray.lzip(left_imgs, right_imgs)

    target_names = np.array(['Different persons', 'Same person'])

    # pack the results as a Bunch instance
    return Bunch(
            pairs=pairs,
            target=target,
            target_names=target_names,
            names=larray.reindex(target_names, target),
            left_imgs=left_imgs,
            right_imgs=right_imgs,
            left_right_imgs=larray.lmap(np.hstack, pairs),
            left_filename=lpaths,
            right_filename=rpaths,
            DESCR="'%s' segment of the LFW pairs dataset" % subset)


#
# Drivers for scikits.data/bin executables
#

def main_fetch():
    """compatibility with bin/datasets_fetch"""
    #TODO: check sys.argv to set funneled=True/False
    check_fetch_lfw()

def main_show():
    from glviewer import glumpy_viewer, command, glumpy
    try:
        import argparse   # new in Python 2.7
        assert sys.argv[1] == 'lfw'
        sys.argv[1:2] = []

        parser = argparse.ArgumentParser(
                description='Show the Labeled Faces in the Wild (lfw) dataset')
        # task
        parser.add_argument('task',
                type=str,
                default='people',
                help='task: "pairs" or "people"')
        # color
        parser.add_argument('--color', action='store_true', dest='color',
                help='load the images in color (default)')
        parser.add_argument('--no-color', action='store_false', dest='color')
        # resize
        parser.add_argument('--resize', type=float, default=1.0,
                help="fraction of original image size")
        # subset
        parser.add_argument('--subset', type=str, default='train',
                help='for "pairs", which subset to load (train/test/10_folds)')

        args = parser.parse_args()
        if args.task == 'people':
            people = load_lfw_people(
                    resize=args.resize,
                    color=args.color,
                    slice_=None)
            n_rows = len(people.imgs)
            print 'n. rows', n_rows
            glumpy_viewer(
                    img_array=people.imgs,
                    arrays_to_print=[people.target, people.names],
                    cmap=glumpy.colormap.Grey)
        elif args.task == 'pairs':
            pairs = load_lfw_pairs(
                    subset=args.subset,
                    resize=args.resize,
                    color=args.color,
                    slice_=None)
            n_rows = len(pairs.left_imgs)
            print 'n. rows', n_rows
            glumpy_viewer(
                    img_array=pairs.left_right_imgs,
                    arrays_to_print=[pairs.target, pairs.names],
                    cmap=glumpy.colormap.Grey,
                    window_shape=(512, 256))

        else:
            raise NotImplementedError(args.task)
    except ImportError:
        logger.warn('no argparse - ignoring arguments')
        # argparse isn't installed, so just show something
        people = load_lfw_people()
        n_rows = len(people.imgs)
        print 'n. rows', n_rows
        glumpy_viewer(
                img_array=people.imgs,
                arrays_to_print=[people.target, people.names])
