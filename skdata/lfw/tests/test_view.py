
from skdata import lfw

def test_view2_smoke_shape():
    for ds_name in lfw.view.DATASET_NAMES:
        view2 = getattr(lfw.view, '%sView2' % ds_name)()
        assert len(view2.x) == 6000
        assert len(view2.y) == 6000
        assert len(view2.x[0]) == 2
        assert view2.x[0][0].shape[:2] == (250, 250)
        if view2.dataset.COLOR:
            view2.x[0][0].shape[-1] == 3
        assert len(view2.splits) == 10
        assert len(view2.splits[0].x) == 5400
        assert len(view2.splits[0].y) == 5400
