
============
Scikits Data
============

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
and this list of dictionaries is referred to as ``self.meta``.

Some datasets are for all intents and purposes *infinite* (i.e. dynamically
generated).  In such cases ``self.meta`` is implemented as lazily-evaluated list.

Sometimes there is meta-data that feels like it applies to an entire dataset
rather than any particular example.  That data goes into a dictionary in
``self.descr``.

Sometimes there is meta-data that is constant across all examples (e.g. image
size).  Such data lives in ``self.meta_const``.  When such an attribute exists,
then every element of `self.meta` will be consistent with it. In other words,
`self.meta[i] == self.meta[i].update(self.meta_const)` is always true.

Datasets often have to download large files from the web. When they do, they
store that bulky data in a dataset-specific subdirectory within `$SCIKITS_DATA`.

Dataset class instances should try to maintain a small footprint. Large
collections of images for example do not live in the dataset instance directly.
The locations of the images should be stored in the dataset, and the images
themselves made available via a lazy-evaluation mechanism. This package provides
the `larray` to help with that, but memory-mapped files and other techniques for
working with large amounts of data are welcome as well.


Development Status
==================

Just starting.

Building on previous work in several projects (scikits.learn, PyLearn, PyLearn2,
PyThor).


Contents
========

The library contains modules and programs.

Modules

  - lfw

Programs

  - dataset-fetch download a dataset


Programs
========

dataset-fetch
-------------

Usage: dataset-fetch <dataset_name>

This program downloads the named dataset into the SCIKIT_LEARN_HOME data directory.
Typically this means downloading from original sources online.

How does this work? It works something like this:

.. code:: python

    exec "datasets.%s" % dataset_name
    exec "datasets.%s.main_fetch()" % dataset_name

So every dataset module has to opt into this mechanism by implementing a global
main_fetch function.
