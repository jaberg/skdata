---
layout: default
title: Scikits.data

section: home

---

**wran·gle**
: (v.) round up, herd, or take charge of (e.g. livestock or just-downloaded data sets)

_skdata_ is a Python library of standard data sets for machine learning experiments.
The modules of `skdata` download data sets, load them as directly as possible
as Python data structures, and provide convenient views for standard machine learning problems.

## Installation

<pre><code>$ git clone https://github.com/jaberg/scikit-data
$ cd scikit-data
$ python setup.py develop</code></pre>


## Goal

Standardize the representation of community benchmark data sets (including large and awkward ones),
and facilitate the development of broadly applicable machine learning algorithm implementations.


## Status

The code of the library is currently usable (and frequently used), but the API
should be considered to be unstable.

Slowly, but surely, the data set modules are being re-factored to match the
"View API"
([docs](https://github.com/jaberg/scikit-data/wiki/View-API),
[code](https://github.com/jaberg/scikit-data/blob/master/skdata/base.py)),
and [add more data set modules](https://github.com/jaberg/scikit-data/wiki/How-to-Create-a-New-Dataset-Module).

