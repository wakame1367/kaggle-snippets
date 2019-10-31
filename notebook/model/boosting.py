import gc

import lightgbm as lgb
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split


def lightgbm_classifier():
    X, y = load_iris(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
    del X, y
    gc.collect()
    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_test = lgb.Dataset(test_x, label=test_y)

    params = {'objective': 'multiclass', 'num_class': 3,
              'verbose': -1}
    clf = lgb.train(params, lgb_train,
                    valid_sets=lgb_test, verbose_eval=False)
    pred_y = clf.predict(test_x)
    print(pred_y)


def lightgbm_regression():
    X, y = load_boston(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
    del X, y
    gc.collect()
    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_test = lgb.Dataset(test_x, label=test_y)

    params = {'objective': 'regression', 'verbose': -1}
    clf = lgb.train(params, lgb_train,
                    valid_sets=lgb_test, verbose_eval=False)
    pred_y = clf.predict(test_x)
    print(pred_y)
