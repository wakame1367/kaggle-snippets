"""
Reference:
https://wakame1367.hatenablog.com/entry/2019/10/29/223844
"""

import os

KAGGLE_ENV_KEYS = {'KAGGLE_KERNEL_INTEGRATIONS', 'KAGGLE_DATA_PROXY_TOKEN',
                   'MPLBACKEND', 'KAGGLE_GYM_DATASET_PATH',
                   'KAGGLE_DATA_PROXY_URL', 'KAGGLE_DATA_PROXY_PROJECT',
                   'TESSERACT_PATH', 'HOSTNAME', 'PYTHONPATH',
                   'KAGGLE_KERNEL_RUN_TYPE', 'JUPYTER_CONFIG_DIR',
                   'PATH', 'LD_LIBRARY_PATH', 'KAGGLE_DATASET_PATH',
                   'MKL_THREADING_LAYER', 'PYTHONUSERBASE', 'LANG', 'PROJ_LIB',
                   'KAGGLE_WORKING_DIR', 'KAGGLE_URL_BASE', 'HOME', 'LC_ALL',
                   'KAGGLE_USER_SECRETS_TOKEN'}


def is_kaggle_kernel() -> bool:
    current_running_kernel_keys = set(os.environ.keys())
    _is_kaggle_kernel = KAGGLE_ENV_KEYS.issubset(current_running_kernel_keys)
    return _is_kaggle_kernel
