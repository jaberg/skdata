===========
scikit-data
===========

Scikits data is a library of datasets for empirical computer science. Lots of
disciplines such as machine learning, natural language processing, and computer
vision have datasets.  This module makes the most popular and standard datasets
(even the big awkward ones) easy to access from Python programs.


Goal
====

Consolidate several overlapping projects:

- scikits.learn.datasets

- pylearn.datasets

- pythor.datasets

Standardize to some extent the representation of large datasets, so that it is
possible to implement standard processing / learning algorithms for those
datasets.


Design
======

Datasets are primarily lists of meta-data, in dictionaries.
Usually one example from the dataset is one element of the list.
The dictionary structures need not necessarily be homogeneous (same keys, same
schema) but often it is natural to make them homogeneous when the examples of the
dataset represent I.I.D. data.  Datasets are typically implemented as classes,
and this list of dictionaries is referred to as ``self.meta``.  Elements of
``self.meta`` should be built of simple data types and be JSON-encodable.  This
opens the door to using various tools to store and access meta-data.

Some datasets are for all intents and purposes *infinite* (i.e. dynamically
generated).  In such cases ``self.meta`` is implemented as lazily-evaluated list.

Sometimes there is meta-data that feels like it applies to an entire dataset
rather than any particular example.  That data goes into a dictionary in
``self.descr``.

Sometimes there is meta-data that is constant across all examples (e.g. image
size).  Such data lives in ``self.meta_const``.  When such an attribute exists,
then every element of ``self.meta`` will be consistent with it. In other words,
``self.meta[i] == self.meta[i].update(self.meta_const)`` is always true.

Datasets often have to download large files from the web. When they do, they
store that bulky data in a dataset-specific subdirectory within ``$SCIKITS_DATA``.

Dataset class instances should try to maintain a small footprint. Large
collections of images for example do not live in the dataset instance directly.
The locations of the images should be stored in the dataset, and the images
themselves made available via a lazy-evaluation mechanism. This package provides
the ``larray`` to help with that, but memory-mapped files and other techniques for
working with large amounts of data are welcome as well.

Many datasets are meant to be used in the context of particular *tasks*.
When that is the case, a dataset class will provide methods that filter
self.meta and return the representation of the data that is most appropriate for
the task.  For example, a classification task method on a small dataset will
return an ``X, y`` pair where ``X`` is numpy ndarray design matrix and ``y`` is a numpy
array of integers.  For a larger dataset, the design matrix might be provided by
a lazily-evaluated proxy object.

Lazy-evaluation is important to the design of data processing pipelines for
large datasets. It decouples the logic of providing access to data from the
logic of efficient caching, which is often task specific.

More design and reference documentation will be written in ``doc/``.
This is already starting with ``doc/dataset.rst``.


Development Status
==================

Just starting.

Building on previous work in several projects (scikits.learn, PyLearn, PyLearn2,
PyThor).


Contents
========

The library contains modules and programs.

Modules:

- toy (Iris, Digits, Diabetes, Linnerud, Boston, SampleImages)
- synthetic (Swiss Roll, S Curve, Friedman{1,2,3}, Madelon, Randlin, Blobs,
  SparseCodedSignal, LowRankMatrix, SparseUncorrelated)
- mnist (MNIST: hand-drawn digit classification dataset)
- lfw (Labeled Faces in the Wild: face recognition and verification)
- pascal (PASCAL Visual Object Classes 2007-2011)
- cifar10 (CIFAR-10 image classification dataset)
- caltech (Caltech101 and Caltech256 Object datasets)
- iicbu (IICBU 2008: Biomedical Image Classification)
- larochelle_etal_2007 (MNIST Variations, Rectangles, and 
    Convex: Image classification tasks with multiple factors of variation)
- pubfig83
.. - rgbd


Programs:

- ``datasets-fetch <dataset>`` download a dataset
- ``datasets-show <dataset>`` visualize a dataset
- ``datasets-erase <dataset>`` erase a downloaded dataset


Programs
========

datasets-fetch
-------------

Usage: ``datasets-fetch <dataset_name>``

This program downloads the named dataset into the ``$SCIKITS_DATA`` directory.
Typically this means downloading from original sources online.

How does this work? It works something like this:

.. code:: python

    exec "skdata.%s" % dataset_name
    exec "skdata.%s.main_fetch()" % dataset_name

So every dataset module has to opt into this mechanism by implementing a global
main_fetch function.
To see more about how sub-modules use this mechanism, grep the code for ``main_fetch``.

datasets-show
-------------

Usage: ``datasets-show <dataset_name>``

This program downloads the named dataset if necessary into the ``$SCIKITS_DATA`` directory,
loads it, and launches a simple GUI program to visualize the elements of the
dataset.
To see more about how sub-modules use this mechanism, grep the code for ``main_show``.

datasets-erase
-------------

Usage: ``datasets-erase <dataset_name>``

This program erases any data cached or downloaded in support of the named dataset.
To see more about how sub-modules use this mechanism, grep the code for ``main_erase``.

