---
layout: default
title: skdata - home

section: home

---

**wran·gle**
: (v.) round up, herd, or take charge of (e.g. livestock or just-downloaded data sets)

_skdata_ is a Python library of standard data sets for machine learning experiments.
The modules of `skdata`
1. _download_ data sets,
2. _load_ them as directly as possible as Python data structures, and
3. _provide protocols_ for machine learning tasks via convenient views.



## Installation

The recommended installation method is to install via pypi with either
`pip install skdata` or `easy_install skdata` (you probably want to
use `pip` if you have it).

If you want to stay up to date with the development tip then use git:

    git clone https://github.com/jaberg/skdata
    ( cd skdata && python setup.py develop )


## Goal

The goal with skdata is to standardize the representation
of community benchmark data sets (including large and awkward ones),
and facilitate the development of broadly applicable machine learning algorithm implementations.
Skdata is meant to interoperate with other Python machine learning software
such as
[sklearn](http://scikit-learn.org/stable/) and [pandas](http://pandas.pydata.org/).


## Status

The code of the library is currently usable (and frequently used), but the API
should be considered to be unstable.

The data set modules are not currently implemented in a standard way, but they
are being re-factored to match the "View API"
([docs](https://github.com/jaberg/skdata/wiki/View-API),
[code](https://github.com/jaberg/skdata/blob/master/skdata/base.py)),
and [add more data set modules](https://github.com/jaberg/skdata/wiki/How-to-Create-a-New-Dataset-Module).

