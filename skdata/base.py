"""
Base classes serving as design documentation.
"""

import numpy as np

class DatasetNotDownloadable(Exception):
    pass

class DatasetNotPresent(Exception):
    pass


class Task(object):
    """
    A Task is the smallest unit of data packaging for training a machine
    learning model.  For different machine learning applications (semantics)
    the attributes are different, but there are some conventions.

    For example:
    semantics='vector_classification'
        - self.x is a matrix-like feature matrix with a row for each example
          and a column for each feature.
        - self.y is a array of labels (any type, but often integer or string)

    semantics='image_classification'
        - self.x is a 4D structure images x height x width x channels
        - self.y is a array of labels (any type, but often integer or string)

    semantics='indexed_vector_classification'
        - self.all_vectors is a matrix (examples x features)
        - self.all_labels is a vector of labels
        - self.idxs is a vector of relevant example positions

    semantics='indexed_image_classification'
        - self.all_images is a 4D structure (images x height x width x channels)
        - self.all_labels is a vector of labels
        - self.idxs is a vector of relevant example positions

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
    # XXX This class is no longer necessary in the View API

    def __init__(self, train, test):
        self.train = train
        self.test = test


class View(object):
    """
    A View is an interpretation of a data set as a standard learning problem.
    """

    def __init__(self, dataset=None):
        """
        dataset: a reference to a low-level object that offers access to the
                 raw data. It is not standardized in any way, and the
                 reference itself is optional.

        """
        self.dataset = dataset

    def protocol(self, algo):
        """
        Return a list of instructions for a learning algorithm.

        An instruction is a 3-tuple of (attr, args, kwargs) such that
        algo.<attr>(*args, **kwargs) can be interpreted by the learning algo
        as a sensible operation, like train a model from some data, or test a
        previously trained model.

        See `LearningAlgo` below for a list of standard instructions that a
        learning algorithm implementation should support, but the protocol is
        left open deliberately so that new View objects can call any method
        necessary on a LearningAlgo, even if it means calling a relatively
        unique method that only particular LearningAlgo implementations
        support.

        """
        raise NotImplementedError()


class LearningAlgo(object):
    """
    A base class for learning algorithms that can be driven by the protocol()
    functions that are sometimes included in View subclasses.

    The idea is that a protocol driver will call these methods in a particular
    order with appropriate tasks, splits, etc. and a subclass of this instance
    will thereby perform an experiment by side effect on `self`.
    """

    def task(self, *args, **kwargs):
        # XXX This is a typo right? Surely there is no reason for a
        # LearningAlgo to have a self.task method...
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


class SemanticsDelegator(LearningAlgo):
    def best_model(self, train, valid=None):
        if valid:
            assert train.semantics == valid.semantics
        return getattr(self, 'best_model_' + train.semantics)(train, valid)

    def loss(self, model, task):
        return getattr(self, 'loss_' + task.semantics)(model, task)


class SklearnClassifier(SemanticsDelegator):
    """
    Implement a LearningAlgo as much as possible in terms of an sklearn
    classifier.

    This class is meant to illustrate how to create an adapter between an
    existing implementation of a machine learning algorithm, and the various
    data sets defined in the skdata library.

    Researchers are encouraged to implement their own Adapter classes
    following the example of this class (i.e. cut & paste this class)
    to measure the statistics they care about when handling the various
    methods (e.g. best_model_vector_classification) and to save those
    statistics to a convenient place. The practice of appending a summary
    dictionary to the lists in self.results has proved to be useful for me,
    but I don't see why it should in general be the right thing for others.


    This class is also used for internal unit testing of Protocol interfaces,
    so it should be free of bit rot.

    """
    def __init__(self, new_model):
        self.new_model = new_model
        self.results = {
            'best_model': [],
            'loss': [],
        }

    def best_model_vector_classification(self, train, valid):
        # TODO: use validation set if not-None
        model = self.new_model()
        print 'SklearnClassifier training on data set of shape', train.x.shape
        model.fit(train.x, train.y)
        model.trained_on = train.name
        self.results['best_model'].append(
            {
                'train_name': train.name,
                'valid_name': valid.name if valid else None,
                'model': model,
            })
        return model

    def loss_vector_classification(self, model, task):
        p = model.predict(task.x)
        err_rate = np.mean(p != task.y)

        self.results['loss'].append(
            {
                'model_trained_on': model.trained_on,
                'predictions': p,
                'err_rate': err_rate,
                'n': len(p),
                'task_name': task.name,
            })

        return err_rate

    @staticmethod
    def _fallback_indexed_vector(self, task):
        return Task(
            name=task.name,
            semantics="vector_classification",
            x=task.all_vectors[task.idxs],
            y=task.all_labels[task.idxs])

    def best_model_indexed_vector_classification(self, train, valid):
        return self.best_model_vector_classification(
            self._fallback_indexed_vector(train),
            self._fallback_indexed_vector(valid))

    def loss_indexed_vector_classification(self, model, task):
        return self.loss_vector_classification(model,
            self._fallback_indexed_vector(task))

    @staticmethod
    def _fallback_indexed_image_task(task):
        if task is None:
            return None
        x = task.all_images[task.idxs]
        y = task.all_labels[task.idxs]
        if 'int' in str(x.dtype):
            x = x.astype('float32') / 255
        else:
            x = x.astype('float32')
        x2d = x.reshape(len(x), -1)
        rval = Task(
            name=task.name,
            semantics="vector_classification",
            x=x2d,
            y=y)
        return rval

    def best_model_indexed_image_classification(self, train, valid):
        return self.best_model_vector_classification(
            self._fallback_indexed_image_task(train),
            self._fallback_indexed_image_task(valid))

    def loss_indexed_image_classification(self, model, task):
        return self.loss_vector_classification(model,
            self._fallback_indexed_image_task(task))


