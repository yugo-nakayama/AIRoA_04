import numpy as np

from shape_ai import CLASSES, classify, make_dataset, synth_mask, train


def test_feature_dataset_is_finite():
    x, y = make_dataset(n_per_class=4, seed=2)
    assert x.shape[0] == len(CLASSES) * 4
    assert np.isfinite(x).all()
    assert set(y) == {0, 1, 2}


def test_small_model_recognizes_clean_shapes(tmp_path):
    bundle, report = train(tmp_path / "model.joblib", n_per_class=120, seed=5)
    rng = np.random.default_rng(99)
    predictions = [classify(bundle, synth_mask(label, rng), tta=True)[0] for label in CLASSES]
    assert predictions == list(CLASSES)
    assert report["accuracy"] > 0.90
