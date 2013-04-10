from skdata.posner_keele.view import PosnerKeele1968E3

import numpy as np

from sklearn.svm import LinearSVC
from skdata.base import SklearnClassifier


def test_protocol(cls=LinearSVC, N=1, show=True, net=None):
    ### run on 36 subjects
    results = {}
    algo = SklearnClassifier(cls)

    pk = PosnerKeele1968E3()
    pk.protocol(algo)

    print cls
    for loss_report in algo.results['loss']:
        print loss_report['task_name'] + \
            (": err = %0.3f" % (loss_report['err_rate']))
    print

    for loss_report in algo.results['loss']:
        task = loss_report['task_name']
        if task not in results: results[task] = []
        results[task].append((loss_report['err_rate'], loss_report['n']))

    stats = {}
    for k, v in results.items():
        p = np.mean([vv[0] for vv in v])
        std = np.sqrt(p*(1-p))
        n = np.sum([vv[1] for vv in v])
        stats[k] = [p, std, n]

    metastats = dict([(k, [np.mean(vi) for vi in v]) for k, v in stats.items()])
    return metastats


