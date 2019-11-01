import numpy as np
import pytest

from notebook.optimizer.optimizer import OptimizedRounderV1, OptimizedRounderV2

size = 10
num_classes = 5
predictions = np.random.randint(num_classes, size=size)
targets = np.random.randint(num_classes, size=size)


@pytest.mark.parametrize("opt_r, data",
                         [
                             (OptimizedRounderV1(), (predictions, targets)),
                             (OptimizedRounderV2(), (predictions, targets))
                         ]
                         )
def test_optimized_rounder(opt_r, data):
    _predictions, _targets = data
    opt_r.fit(X=_predictions, y=_targets)
    coefficients = opt_r.coefficients()
    assert len(coefficients) == num_classes - 1
