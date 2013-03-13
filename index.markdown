---
layout: default
title: skdata - home

section: home

---

**wran·gle**
: (v.) round up, herd, or take charge of (e.g. livestock or just-downloaded data sets)

**_skdata_** is a Python library of standard data sets for machine learning experiments.
The modules of `skdata`
1. **download** data sets,
2. **load** them as directly as possible as Python data structures, and
3. **provide protocols** for machine learning tasks via convenient views.

## Gist

To demonstrate the full system, here's how you would evaluate a Support Vector
Machine (scikit-learn's LinearSVC) as a classification model for the UCI
"Iris" data set:

<code class="brush: python;"><pre># Create a suitable view of the Iris data set.
# (For larger data sets, this can trigger a download the first time)
from skdata.iris.view import KfoldClassification
iris_view = KfoldClassification(5)

# Create a learning algorithm based on scikit-learn's LinearSVC
# that will be driven by commands the `iris_view` object.
from sklearn.svm import LinearSVC
from skdata.base import SklearnClassifier
learning_algo = SklearnClassifier(LinearSVC)

# Drive the learning algorithm from the data set view object.
# (An iterator interface is sometimes also be available,
#  so you don't have to give up control flow completely.)
iris_view.protocol(learning_algo)

# The learning algorithm keeps track of what it did when under
# control of the iris_view object. This base example is useful for
# internal testing and demonstration. Use a custom learning algorithm
# to track and save the statistics you need.
for loss_report in algo.results['loss']:
    print loss_report['task_name'] + \
        (": err = %0.3f" % (loss_report['err_rate']))
</pre></code>

Note that you can also use the `skdata.iris.dataset` module to get raw
un-standardized access to the Iris data set via Python objects.  This is the
pattern used throughout `skdata`: dataset submodules give raw access,
and view submodules implement standardized views.

## Installation

The recommended installation method is to install via pypi with either
`pip install skdata` or `easy_install skdata` (you probably want to
use `pip` if you have it).

If you want to stay up to date with the development tip then use git:

<code class="brush: bash;"><pre>git clone https://github.com/jaberg/skdata
( cd skdata && python setup.py develop )
</pre></code>


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

