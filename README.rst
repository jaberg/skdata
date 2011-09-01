
============
Scikits Data
============

Scikits data is a library of datasets for empirical computer science: lots of
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
dataset represent I.I.D. data.  This list of dictionaries is referred to as
".meta"


Everything stored in dataset-specific subdirs of `SCIKIT_LEARN_HOME/`.

Lazy evaluation and proxy objects used to delay expensive computations, avoid
loading large data structures.


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
