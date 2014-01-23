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

Here's a [list of data sets](https://github.com/jaberg/skdata/wiki/Data-Set-Modules) that skdata provides.

## Gist

Here's how skdata helps you evaluate an SVM (e.g. [scikit-learn](http://scikit-learn.org)'s
[LinearSVC](http://scikit-learn.org/stable/modules/svm.html)) as a classifier for the
UCI "[Iris](http://archive.ics.uci.edu/ml/datasets/Iris)" data set:

<!-- pygmentize -f index_demo.py -->
<div class="highlight"><pre><span class="c"># Create a suitable view of the Iris data set.</span>
<span class="c"># (For larger data sets, this can trigger a download the first time)</span>
<span class="kn">from</span> <span class="nn">skdata.iris.view</span> <span class="kn">import</span> <span class="n">KfoldClassification</span>
<span class="n">iris_view</span> <span class="o">=</span> <span class="n">KfoldClassification</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>

<span class="c"># Create a learning algorithm based on scikit-learn&#39;s LinearSVC</span>
<span class="c"># that will be driven by commands the `iris_view` object.</span>
<span class="kn">from</span> <span class="nn">sklearn.svm</span> <span class="kn">import</span> <span class="n">LinearSVC</span>
<span class="kn">from</span> <span class="nn">skdata.base</span> <span class="kn">import</span> <span class="n">SklearnClassifier</span>
<span class="n">learning_algo</span> <span class="o">=</span> <span class="n">SklearnClassifier</span><span class="p">(</span><span class="n">LinearSVC</span><span class="p">)</span>

<span class="c"># Drive the learning algorithm from the data set view object.</span>
<span class="c"># (An iterator interface is sometimes also be available,</span>
<span class="c">#  so you don&#39;t have to give up control flow completely.)</span>
<span class="n">iris_view</span><span class="o">.</span><span class="n">protocol</span><span class="p">(</span><span class="n">learning_algo</span><span class="p">)</span>

<span class="c"># The learning algorithm keeps track of what it did when under</span>
<span class="c"># control of the iris_view object. This base example is useful for</span>
<span class="c"># internal testing and demonstration. Use a custom learning algorithm</span>
<span class="c"># to track and save the statistics you need.</span>
<span class="k">for</span> <span class="n">loss_report</span> <span class="ow">in</span> <span class="n">algo</span><span class="o">.</span><span class="n">results</span><span class="p">[</span><span class="s">&#39;loss&#39;</span><span class="p">]:</span>
    <span class="k">print</span> <span class="n">loss_report</span><span class="p">[</span><span class="s">&#39;task_name&#39;</span><span class="p">]</span> <span class="o">+</span> \
        <span class="p">(</span><span class="s">&quot;: err = </span><span class="si">%0.3f</span><span class="s">&quot;</span> <span class="o">%</span> <span class="p">(</span><span class="n">loss_report</span><span class="p">[</span><span class="s">&#39;err_rate&#39;</span><span class="p">]))</span>
</pre></div>

Note that you can also use the `skdata.iris.dataset` module to get raw
un-standardized access to the Iris data set via Python objects.
This is an skdata convention: `dataset` submodules give raw access,
and `view` submodules implement standardized views and protocols.


## Installation

The recommended installation method is via pypi with either
`pip install skdata` or `easy_install skdata` (you probably want to
use `pip` if you have it).

If you want to stay up to date with the development tip then use git:

<!-- pygmentize -f index_git.sh -->
<div class="highlight"><pre>git clone https://github.com/jaberg/skdata <span class="se">\</span>
<span class="o">&amp;&amp;</span> <span class="o">(</span> <span class="nb">cd </span>skdata python <span class="o">&amp;&amp;</span> setup.py develop <span class="o">)</span>
</pre></div>


## Documentation

Documentation is maintained on the [skdata wiki](https://github.com/jaberg/skdata/wiki/).



