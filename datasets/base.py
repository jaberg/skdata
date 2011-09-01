"""
Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import os
import csv
import shutil
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os import listdir
from os import makedirs

import numpy as np

from .utils import check_random_state

def load_diabetes():
    """Load and return the diabetes dataset (regression).

    Return
    ------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn and 'target', the labels for each
        sample.
    """
    base_dir = join(dirname(__file__), 'data')
    data = np.loadtxt(join(base_dir, 'diabetes_data.csv.gz'))
    target = np.loadtxt(join(base_dir, 'diabetes_target.csv.gz'))
    return Bunch(data=data, target=target)

def load_linnerud():
    """Load and return the linnerud dataset (multivariate regression).

    Return
    ------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data_exercise' and 'data_physiological', the two multivariate
        datasets, as well as 'header_exercise' and
        'header_physiological', the corresponding headers.
    """
    base_dir = join(dirname(__file__), 'data/')
    # Read data
    data_exercise = np.loadtxt(base_dir + 'linnerud_exercise.csv', skiprows=1)
    data_physiological = np.loadtxt(base_dir + 'linnerud_physiological.csv',
                                    skiprows=1)
    # Read header
    f = open(base_dir + 'linnerud_exercise.csv')
    header_exercise = f.readline().split()
    f.close()
    f = open(base_dir + 'linnerud_physiological.csv')
    header_physiological = f.readline().split()
    f.close()
    fdescr = open(dirname(__file__) + '/descr/linnerud.rst')

    return Bunch(data_exercise=data_exercise, header_exercise=header_exercise,
                 data_physiological=data_physiological,
                 header_physiological=header_physiological,
                 DESCR=fdescr.read())

def load_boston():
    """Load and return the boston house-prices dataset (regression).

    Return
    ------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, and 'DESCR', the
        full description of the dataset.

    Examples
    --------
    >>> from scikits.learn.datasets import load_boston
    >>> data = load_boston()
    """
    module_path = dirname(__file__)
    data_file = csv.reader(open(join(module_path, 'data',
                                     'boston_house_prices.csv')))
    fdescr = open(join(module_path, 'descr', 'boston_house_prices.rst'))
    temp = data_file.next()
    n_samples = int(temp[0])
    n_features = int(temp[1])
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))
    temp = data_file.next()  # names of features
    feature_names = np.array(temp)

    for i, d in enumerate(data_file):
        data[i] = np.asanyarray(d[:-1], dtype=np.float)
        target[i] = np.asanyarray(d[-1], dtype=np.float)

    return Bunch(data=data,
                 target=target,
                 feature_names=feature_names,
                 DESCR=fdescr.read())

def load_sample_images():
    """Load sample images for image manipulation.

    Return
    ------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, `images`, the images corresponding
        to each sample, 'target', the classification labels for each
        sample, 'target_names', the meaning of the labels, and 'DESCR',
        the full description of the dataset.

    Examples
    --------
    To load the data and visualize the images::

    >>> from scikits.learn.datasets import load_sample_images
    >>> dataset = load_sample_images()
    >>> len(dataset.images)
    2
    >>> first_img_data = dataset.images[0]
    >>> first_img_data.shape  # height, width, channels
    (427, 640, 3)
    >>> first_img_data.dtype
    dtype('uint8')

    >>> # import pylab as pl
    >>> # pl.gray()
    >>> # pl.matshow(dataset.images[0]) # Visualize the first image
    >>> # pl.show()
    """
    # Try to import imread from scipy. We do this lazily here to prevent
    # this module from depending on PIL.
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL)"
                          "is required to load data from jpeg files")
    module_path = join(dirname(__file__), "images")
    descr = open(join(module_path, 'README.txt')).read()
    filenames = [join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    images = [imread(filename) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)

def load_sample_image(image_name):
    """Load the numpy array of a single sample image

    >>> china = load_sample_image('china.jpg')
    >>> china.dtype
    dtype('uint8')
    >>> china.shape
    (427, 640, 3)

    >>> flower = load_sample_image('flower.jpg')
    >>> flower.dtype
    dtype('uint8')
    >>> flower.shape
    (427, 640, 3)
    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]
