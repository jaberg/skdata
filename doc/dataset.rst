
=======
Dataset
=======

A class with attributes and methods.

Database API
============

meta
----

Required.

Attribute.
Type: list-like of dict.

If possible, the dict elements should be JSON-encodable, so that meta can be
hosted and served by a standard document database.


meta_const
----------

Optional.

Attribute.
Type: dict.


descr
-----

Optional.

Attribute.
Type: dict.


Task API
========

For the convenience of users just trying to do the normal thing with a dataset,
data set classes are encouraged to adhere to the following protocol as much as
possible.  Datasets should implement these methods in terms of the self.meta,
self.meta_const, and self.descr so that they still work if .meta is implemented
with a document database.


regression_task
---------------

Optional.

Method taking no required arguments.
Returns pair of 2-d ndarray-like X, Y.
Task is to predict Y from X, minimizing MSE.


classification_task
-------------------

Optional.

Method taking no required arguments.
Returns pair X, y.
X is 2-d floating point ndarray-like design matrix of features.
y is 1-d integer ndarray-like of labels.
Task is to predict y from X, minimizing mean cross-entropy.


clustering_task
---------------

Optional.

Method taking no required arguments.
Returns X, a 2-d floating point ndarray-like design matrix.


factorization_task
------------------

Optional.

Method taking no required arguments.
Returns X, a 2-d floating point ndarray-like design matrix.


matrix_completion_task
----------------------

Optional.

Method taking no required arguments.
Returns X, Y.
X and Y are both 2-d floating point sparse matrix, in which implicit elements
carry the semantics of being unobserved, rather than zero.
X and Y do not overlap.
The task is to predict the observations in Y from the observations in X.


Scripting API
=============

There are standalone programs bin/datasets-fetch and bin/datasets-show.
They work based on command-line syntax like this:

.. code-block:: bash

    $ datasets-fetch lfw.Original

The string appearing after the command ``lfw.Original`` is used to import a
symbol called ``datasets.lfw.Original.main_fetch``.  The way this works is that
the string is split on the periods, and imports are attempted from the left
until an ImportError is encountered.  Then the entire string is used as a symbol
name. If that works then datasets-fetch calls that symbol with no arguments.
If that fails (e.g. with AttributeError) then datasets-fetch prints an error
message and exits.  datasets-show works similar, but looks for the trailing symbol
``main_show`` instead of ``main_fetch``.

main_fetch
----------

Optional.

Type: classmethod, staticmethod, or module method.
This method should download the dataset, if that makes sense for the dataset.
It should download the dataset to a subdirectory of the cache folder reported by
``get_data_home()`` in ``data_home.py``.

main_show
---------

Optional.

Type: classmethod, staticmethod, or module method.
This method should visualize the dataset.  This is generally interesting, gives
intuition for what algorithms would make sense, and also convinces the user that
the dataset was downloaded properly.
