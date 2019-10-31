from notebook.model.boosting import (lightgbm_classifier, lightgbm_regression)


def test_lightgbm_classifier():
    num_class = 3
    predicts = lightgbm_classifier()
    assert predicts.shape[1] == num_class


def test_lightgbm_regression():
    predicts = lightgbm_regression()
    assert predicts
