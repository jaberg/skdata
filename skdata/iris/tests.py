
from sklearn.svm import LinearSVC
from skdata.iris.view import KfoldClassification
from skdata.base import SklearnClassifier


def test_protocol(cls=LinearSVC, N=1, show=True, net=None):
    ### run on 36 subjects
    algo = SklearnClassifier(cls)

    pk = KfoldClassification(4)
    mean_test_error = pk.protocol(algo)

    assert len(algo.results['loss']) == 4
    assert len(algo.results['best_model']) == 4

    print cls
    for loss_report in algo.results['loss']:
        print loss_report['task_name'] + \
            (": err = %0.3f" % (loss_report['err_rate']))

    assert mean_test_error < 0.1

