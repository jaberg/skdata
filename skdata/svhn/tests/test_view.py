from skdata import svhn


def test_view1_smoke_shape():
    ds = svhn.view.CroppedDigitsStratifiedKFoldView1(k=2)
    assert len(ds.splits) == 2
    assert ds.splits[0].train.x.shape == (36628, 32, 32, 3)
    assert ds.splits[0].train.y.shape == (36628,)
    assert ds.splits[1].train.x.shape == (36629, 32, 32, 3) 
    assert ds.splits[1].train.y.shape == (36629,)


def test_view2_smoke_shape():
    ds = svhn.view.CroppedDigitsView2()
    assert len(ds.splits) == 1
    assert len(ds.splits[0].train.x) == 73257
    assert len(ds.splits[0].train.y) == 73257
    assert ds.splits[0].train.x[0].shape == (32, 32, 3)
