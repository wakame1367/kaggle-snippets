import numpy as np
import pytest

from notebook.metrics.qwk import quad_kappa, qwk3


@pytest.fixture
def get_test_data():
    np.random.seed(0)
    size = 10
    a = np.random.randint(0, 4, size)
    p = np.random.randint(0, 4, size)
    return a, p


def test_quad_kappa(get_test_data):
    result = quad_kappa(*get_test_data)
    assert isinstance(result, float)


def test_qwk3(get_test_data):
    result = qwk3(*get_test_data)
    assert isinstance(result, float)
