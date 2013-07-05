from unittest import TestCase
from functools import partial

import nose
import numpy as np
try:
    import sklearn.svm
except ImportError:
    pass

from dataset import KaggleFacialExpression
from view import ContestCrossValid
from ..base import DatasetNotPresent
from ..base import SklearnClassifier

ds = KaggleFacialExpression()
try:
    ds.fetch()
    skip_all = False
except DatasetNotPresent:
    skip_all = True

class TestKFE(TestCase):

    def setUp(self):
        if skip_all:
            raise nose.SkipTest()

    def test_len_train(self):
        n = len([mi for mi in ds.meta
            if mi['file'] == 'train.csv'])
        assert n == KaggleFacialExpression.N_TRAIN, n

    def test_len_test(self):
        n = len([mi for mi in ds.meta
            if mi['file'] == 'test.csv'])
        assert n == KaggleFacialExpression.N_TEST, n


class TestContestXV(TestCase):
    def setUp(self):
        if skip_all:
            raise nose.SkipTest()
        self.xv = ContestCrossValid(
                n_train=ContestCrossValid.max_n_train - 7500,
                n_valid=7500,
                ds=ds)

    def test_protocol_smoke(self):
        # -- smoke test that it just runs
        class Algo(object):
            def best_model(algo_self, train, valid=None):
                # -- all training labels should be legit
                assert np.all(train.all_labels[train.idxs] < 7)
                assert np.all(train.all_labels[train.idxs] >= 0)

                # -- N.B. test labels might be unknown, and 
                #    replaced with dummy ones which may or may
                #    not be in range(7)
                return None

            def loss(algo_self, model, task):
                return 1.0

        algo = Algo()
        self.xv.protocol(algo)

    def test_protocol_svm(self):
        if 'sklearn' not in globals():
            raise nose.SkipTest()
        self.xv = ContestCrossValid(
                n_train=200,
                n_valid=100,
                ds=ds)
        algo = SklearnClassifier(
            partial(sklearn.svm.SVC, kernel='linear'))
        self.xv.protocol(algo)
        print algo.results

