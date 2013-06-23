from functools import partial
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skdata.base import SklearnClassifier
from skdata.socrata.austin.restaurant_inspection.view \
        import LocationClassification5

def test_location_prediction():
    K = 10
    lp = LocationClassification5(K=K)
    algo = SklearnClassifier(
        partial(DecisionTreeClassifier,
            max_depth=1))

    mean_test_error = lp.protocol(algo)

    assert len(algo.results['loss']) == K
    assert len(algo.results['best_model']) == K

    for loss_report in algo.results['loss']:
        print loss_report['task_name'] + \
            (": err = %0.3f" % (loss_report['err_rate']))

    print 'mean test error:', mean_test_error

    # -- the dataset changes with each query potentially, and
    #    for sure changes with time, so don't assert anything too specific about
    #    accuracy.
    #
    #    FWIW, June 22, 2013, I was seeing error like around .48

