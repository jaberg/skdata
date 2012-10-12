"""
Base classes serving as design documentation.
"""


class Task(object):
    """
    A Task is the smallest unit of data packaging for training a machine
    learning model.  For different machine learning applications (semantics)
    the attributes are different, but there are some conventions.

    For example:
    'vector classification'
        - self.x is a matrix-like feature matrix with a row for each example
          and a column for each feature.
        - self.y is a array of labels (any type, but often integer or string)

    'image classification'
        - self.x is a 4D structure images x height x width x channels
        - self.y is a array of labels (any type, but often integer or string)

    The design taken in skdata is that each data set view file defines

    * a semantics object (a string in the examples above) that uniquely
      *identifies* what a learning algorithm is supposed to do with the Task,
      and

    * documentation to *describe* to the user what a learning algorithm is
      supposed to do with the Task.

    As library designers, it is our hope that data set authors can re-use each
    others' semantics as much as possible, so that learning algorithms are
    more portable between tasks.

    """
    def __init__(self, semantics=None, name=None, **kwargs):
        self.semantics = semantics
        self.name = name
        self.__dict__.update(kwargs)


class Split(object):
    """
    A Split is a (train, test) pair of Tasks with no common examples.

    This class is used in cross-validation to select / learn parameters
    based on the `train` task, and then to evaluate them on the `valid` task.
    """

    def __init__(self, train, test):
        self.train = train
        self.test = test


class View(object):
    """
    A View is a set of splits as in K-fold for measuring a cross-validation
    error.
    """

    def __init__(self, splits, dataset=None):
        """
        splits: list of Split objects, not generally independent of one
                another.

        dataset: a reference to a low-level object that offers access to the
                 raw data. It is not standardized in any way, and the
                 reference itself is optional.

        """
        self.splits = splits
        self.dataset = dataset


class LearningAlgo(object):
    """
    A base class for learning algorithms that can be driven by the protocol()
    functions that are sometimes included in View subclasses.

    The idea is that a protocol driver will call these methods in a particular
    order with appropriate tasks, splits, etc. and a subclass of this instance
    will thereby perform an experiment by side effect on `self`.
    """

    def task(self, *args, **kwargs):
        return Task(*args, **kwargs)

    def best_model(self, train, valid=None, return_promising=False):
        """
        Train a model from task `train` optionally optimizing for
        cross-validated performance on `valid`.

        If `return_promising` is False, this function returns a tuple:

            (model, train_error, valid_error)

        In which
            model is an opaque model for the task,
            train_error is a scalar loss criterion on the training task
            valid_error is a scalar loss criterion on the validation task.

        If `return_promising` is True, this function returns

            (model, train_error, valid_error, promising)

        The `promising` term is a boolean flag indicating whether the model
        seemed to work (1) or if it appeared to be degenerate (0).

        """
        raise NotImplementedError('implement me')

    def loss(self, model, task):
        """
        Return scalar-valued training criterion of `model` on `task`.

        This function can modify `self` but it should not semantically modify
        `model` or `task`.
        """
        raise NotImplementedError('implement me')

    # -- as an example of weird methods an algo might be required to implement
    #    to accommodate bizarre protocols, see this one, which is required by
    #    LFW.  Generally there is no need for this base class to list such
    #    special-case functions.
    def retrain_classifier(self, model, train, valid=None):
        """
        To the extent that `model` includes a feature extractor that is distinct from
        a classifier, re-train the classifier only. This unusual step is
        required in the original View1 / View2 LFW protocol. It is included
        here as encouragement to add dataset-specific steps in LearningAlgo subclasses.
        """
        raise NotImplementedError('implement me')


    def forget_task(self, task_name):
        """
        Signal that it is OK to delete any features / statistics etc related
        specifically to task `task_name`.  This can be safely ignored
        for small data sets but deleting such intermediate results can
        be crucial to keeping memory use under control.
        """
        pass

