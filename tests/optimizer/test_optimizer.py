import numpy as np
import pytest

from notebook.optimizer.optimizer import OptimizedRounderV1, OptimizedRounderV2


@pytest.fixture
def get_test_data():
    size = 10
    num_classes = 5
    predictions = np.random.randint(num_classes, size=size)
    targets = np.random.randint(num_classes, size=size)
    return predictions, targets


@pytest.mark.parametrize("opt_r",
                         [
                             OptimizedRounderV1(),
                             OptimizedRounderV2()
                         ]
                         )
def test_optimized_rounder(opt_r, get_test_data):
    num_classes = 5
    _predictions, _targets = get_test_data
    opt_r.fit(X=_predictions, y=_targets)
    coefficients = opt_r.coefficients()
    assert len(coefficients) == num_classes - 1
