from datetime import date, timedelta

import pandas as pd
import pytest

from notebook.feature.time_feature import extract_time_features, jp_holidays


@pytest.fixture
def get_test_data():
    today = date.today()
    date_time = []
    time_col = 'datetime'
    for i in range(1000):
        date_time.append(today + timedelta(days=i))
    df = pd.DataFrame({
        time_col: date_time
    })
    return df, time_col


def test_extract_time_features(get_test_data):
    df, time_col = get_test_data
    exp_cols = {time_col, 'date', 'month', 'hour', 'dayofweek'}
    result = extract_time_features(df, time_col)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.tolist()).issubset(exp_cols)


def test_jp_holidays(get_test_data):
    df, time_col = get_test_data
    exp_cols = {time_col, 'is_holiday'}
    result = jp_holidays(df, time_col)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.tolist()).issubset(exp_cols)
