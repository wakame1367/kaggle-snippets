import logging

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, logger: logging.RootLogger = None,
                     level: int = logging.DEBUG) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
    Reference:
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    https://amalog.hateblo.jp/entry/kaggle-snippets#DataFrame%E3%81%AE%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84%E3%81%99%E3%82%8B
    """
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(
                        np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_(
        'Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
