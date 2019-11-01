def multi_label_f_score(y_true, y_pred):
    """
    Reference:
    https://www.kaggle.com/onodera/multilabel-fscore
    https://github.com/KazukiOnodera/Instacart
    :return:

    Examples
    --------
    >>> y_true, y_pred = [1, 2, 3], [2, 3]
    >>> multi_label_f_score(y_true, y_pred)
    0.8
    >>> y_true, y_pred = [None], [2, None]
    >>> round(multi_label_f_score(y_true, y_pred), 3)
    0.667
    >>> y_true, y_pred = [4, 5, 6, 7], [2, 4, 8, 9]
    >>> multi_label_f_score(y_true, y_pred)
    0.25
    """
    y_true, y_pred = set(y_true), set(y_pred)
    precision = sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    recall = sum([1 for i in y_true if i in y_pred]) / len(y_true)

    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)
