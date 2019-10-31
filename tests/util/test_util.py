from notebook.util.is_kaggle_kernel import is_kaggle_kernel


def test_is_kaggle_kernel():
    is_kaggle = is_kaggle_kernel()
    assert not is_kaggle
