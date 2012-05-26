---
layout: default
title: Scikits.data

section: home

---

**wran·gle**
: (v.) round up, herd, or take charge of (e.g. livestock or just-downloaded datasets)

The `skdata` project is Python library of standard data sets for machine learning experiments.
The modules of `skdata` download data sets, load them as directly as possible
as Python data structures, and where it makes sense --
provide convenient views on those data for standard machine learning problems.

The `skdata` project is a work in progress --
we're working on some [proto-documentation](https://github.com/jaberg/scikit-data/wiki)
of the shape of things to come, which is closer in some cases than others to
how the library currently works.

_Status_: The code of the library is currently usable (and frequently used), but the API
should be considered to be unstable. The plan is to refactor the current data set modules
to match the new "View API"
([docs](https://github.com/jaberg/scikit-data/wiki/View-API),
[code](https://github.com/jaberg/scikit-data/blob/master/skdata/base.py)),
and to continue to [add more data set
modules](https://github.com/jaberg/scikit-data/wiki/How-to-Create-a-New-Dataset-Module).

