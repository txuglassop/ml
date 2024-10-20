import pandas as pd
import xgboost as xgb

from sklearn import linear_model, metrics, preprocessing, ensemble

def run_fold_logistic(data, fold):
    """
    Function that performs validation set on a specific fold, training w/ logistic regression
    performs one-hot encoding and fills in NaN as new categories

    Parameters:
        data - pandas df with the kfold column for cv
        fold - which fold to leave out (validation set)

    Returns:
        nothing
    """

    # get features
    features = [
      f for f in data.columns if f not in ('id', 'target', 'kfold')
    ] 

    for col in features:
      data.loc[:,col] = data[col].astype(str).fillna("NONE")

    train = data[data.kfold != fold].reset_index(drop=True)
    valid = data[data.kfold == fold].reset_index(drop=True)

    # initialise OHE
    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
       [train[features], valid[features]], axis =0
    )
    ohe.fit(full_data[features])

    X_train = ohe.transform(train[features])
    X_valid = ohe.transform(valid[features])

    model = linear_model.LogisticRegression()

    model.fit(X_train, train.target.values)

    preds = model.predict_log_proba(X_valid)[:,1]
    auc = metrics.roc_auc_score(valid.target.values, preds)

    print(f'Fold {fold}: AUC: {auc}')

def run_fold_rf(data, fold):
    """
    Function that performs validation set on a specific fold
    performs label encoding and trains a random forest

    Parameters:
        data - pandas df with the kfold column for cv
        fold - which fold to leave out (validation set)

    Returns:
        nothing
    """

    # get features
    features = [
      f for f in data.columns if f not in ('id', 'target', 'kfold')
    ] 

    for col in features:
      data.loc[:,col] = data[col].astype(str).fillna("NONE")

    # initialise OHE
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(data[col])
        data.loc[:,col] = lbl.transform(data[col])

    train = data[data.kfold != fold].reset_index(drop=True)
    valid = data[data.kfold == fold].reset_index(drop=True)

    X_train = train[features].values
    X_valid = valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs=-1)

    train['target'] = train['target'].astype(int)

    model.fit(X_train, train.target.values)

    preds = model.predict_proba(X_valid)[:,1]
    auc = metrics.roc_auc_score(valid.target.values, preds)

    print(f'Fold {fold}: AUC: {auc}')

def run_fold_xgb(data, fold):
    """
    Function that performs validation set on a specific fold
    performs label encoding and trains a random forest

    Parameters:
        data - pandas df with the kfold column for cv
        fold - which fold to leave out (validation set)

    Returns:
        nothing
    """

    # get features
    features = [
      f for f in data.columns if f not in ('id', 'target', 'kfold')
    ] 

    for col in features:
      data.loc[:,col] = data[col].astype(str).fillna("NONE")

    # initialise OHE
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(data[col])
        data.loc[:,col] = lbl.transform(data[col])

    train = data[data.kfold != fold].reset_index(drop=True)
    valid = data[data.kfold == fold].reset_index(drop=True)

    X_train = train[features].values
    X_valid = valid[features].values

    model = xgb.XGBClassifier(
       n_jobs=-1,
       max_depth=7,
       n_estimators=200
    )

    model.fit(X_train, train.target.values)

    preds = model.predict_proba(X_valid)[:,1]
    auc = metrics.roc_auc_score(valid.target.values, preds)

    print(f'Fold {fold}: AUC: {auc}')