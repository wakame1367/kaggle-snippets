import pandas as pd


def extract_time_features(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df['date'] = df[time_col].dt.date
    df['month'] = df[time_col].dt.month
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    return df
