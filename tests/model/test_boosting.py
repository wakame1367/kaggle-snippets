from notebook.model.boosting import (lightgbm_classifier, lightgbm_regression)


def test_lightgbm_classifier():
    predicts = lightgbm_classifier
    assert predicts


def test_lightgbm_regression():
    predicts = lightgbm_regression()
    assert predicts
