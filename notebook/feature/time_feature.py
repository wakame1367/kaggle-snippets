import jpholiday
import pandas as pd


def extract_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df[time_col] = pd.to_datetime(df[time_col])
    df['date'] = df[time_col].dt.date
    df['month'] = df[time_col].dt.month
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    return df


def jp_holidays(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Reference: https://upura.hatenablog.com/entry/2018/12/21/070000
    :return:
    """
    df['is_holiday'] = df[time_col].map(jpholiday.is_holiday).astype(int)
    return df
