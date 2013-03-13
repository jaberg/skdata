
# Create a suitable view of the Iris data set.
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
