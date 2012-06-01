
from .views import KfoldClassification

def test_view_kfc():
    kfc = KfoldClassification(4)
    assert len(kfc.splits) == 4

    # fail if the following symbols are undefined
    kfc.splits[0].train.x
    kfc.splits[0].train.y
    kfc.splits[3].test.x
    kfc.splits[3].test.y

    assert isinstance(kfc.splits[2].test.y[0], str)


def test_view_kfc_intlabels():
    kfc = KfoldClassification(2, y_as_int=True)
    assert isinstance(kfc.splits[0].train.y[-1], int)

