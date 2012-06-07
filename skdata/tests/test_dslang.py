from ..dslang import Visitor
from ..iris.views import KfoldClassification
import sklearn.linear_model


class SVMAlgo(Visitor):
    def __init__(self, model_factory):
        self.model_factory = model_factory

    def on_BestModel(self, node, memo):
        train = self.evaluate(node.split, memo)
        model = self.model_factory()
        model.fit(train.x, train.y)
        return model

    def on_Score(self, node, memo):
        model = self.evaluate(node.model, memo)
        task = self.evaluate(node.task, memo)
        y_pred = model.predict(task.x)
        return (y_pred == task.y).mean()

def test_dslang():
    kfc = KfoldClassification(2, y_as_int=True)

    def new_model():
        return sklearn.linear_model.SGDClassifier()

    memo = {}
    dsl = kfc.dsl
    SVMAlgo(new_model).evaluate(dsl, memo)

    print 'final score', memo[dsl]


