---
layout: default
title: Scikits.data

section: home

---

Scikits.data
============

**wran·gle**
: (v.) round up, herd, or take charge of (livestock or just-downloaded datasets)

Scikist.data is Python library for presenting standard datasets as usable Python
objects.  Datasets are used all the time in e.g. machine learning, computer vision,
music information retrieval, natural language processing.  They come in all
kinds of shapes and sizes, such as:
* tarballs of xml files and subdirectories of carefully named image files,
* Python pickle files,
* non-standard binary dumps.

Scikits.data does two essential things:
1. **downloads standard datasets**
2. **turns them into usable Python objects**

After that it's up to you.  You can use those Python objects directly,
or corale that data toward other libraries' data structures such as
[pytables](TODOpytables), [numpy](http://www.scipy.org/numpy),
[mongo](http://www.mongodb.org),
[pandas](TODOpandas), or any other SQL or no-SQL database.

Scikits.data also provides a higher-level interface: **tasks**.
Many datasets are
distributed for the purpose of evaluating learning algorithms on particular
tasks.
Tasks are strictly specified protocols for passing structured data around.
For example, there is a classification task protocol, and a regression task
protocol.
If a learning algorithm is written against a task protocol, it will generally
work on all of the datasets that provide that task.
A dataset can be used to define multiple kinds of tasks (e.g. classification,
regression), and even many versions of a particular kind of task (e.g.
multi-class classification can be presented as several
instances of binary classification)
[Learn more about the task API.](TODO)

Scikits.data is designed to give access to large and possibly infinite data
sets.
also provides some 
