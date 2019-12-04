import category_encoders as cat_enc


def target_encoding_with_catboost(cols, x, y):
    encoder = cat_enc.CatBoostEncoder(cols=cols)
    encoder.fit(x, y)
    return encoder.transform(x)
