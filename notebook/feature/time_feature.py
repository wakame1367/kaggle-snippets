"""
jpholiday:
https://github.com/Lalcs/jpholiday
"""
import jpholiday
import numpy as np
import pandas as pd


def extract_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col])
    df['date'] = df[time_col].dt.date
    df['month'] = df[time_col].dt.month
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in (5, 6) else 0)
    return df


def jp_holidays(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Reference: https://upura.hatenablog.com/entry/2018/12/21/070000
    :return:
    """
    df['is_holiday'] = df[time_col].map(jpholiday.is_holiday).astype(int)
    return df


def make_harmonic_features(value: int, period: int = 24) -> (float, float):
    """　周期的/循環性のある値をsin/cosに変換する
    周期的/循環性のある値とは例えば一日の時間(24時間)や日付(365日)
    Reference:
    https://www.kaggle.com/kashnitsky/topic-6-feature-engineering-and-feature-selection
    https://qiita.com/shimopino/items/4ef78aa589e43f315113
    
    :param value:
    :param period:
    :return:

    >>> one_day_time = list((range(1, 24+1)))
    >>> make_harmonic_features(one_day_time[0], period=24)
    (0.9659258262890683, 0.25881904510252074)
    >>> make_harmonic_features(one_day_time[11], period=24)
    (-1.0, 1.2246467991473532e-16)
    >>> make_harmonic_features(one_day_time[-1], period=24)
    (1.0, -2.4492935982947064e-16)
    """
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)
