import numpy as np
import pandas as pd
import joblib

from sys import argv, exit
from sklearn import tree, metrics, datasets, ensemble

from init_folds import init_folds
from model_dispatcher import models

# could add a config.py file with directorie and what not for project etc.

def run_fold_regression(data, fold, target, model):
    """
    function that performs hold out validation, holding out specified fold. 
    trains using a tree model, uses MAE as metric. saves model into directory and prints metric

    parameters:
        data: pandas df with 'kfold' column
        param: fold - specify which fold to hold out (test set)
        target: specifying the name of the target variable
        model: specify the model, using the model_dispatcher

    returns:
        nothing 
    """
    train = data[data.kfold != fold].reset_index(drop=True)
    test = data[data.kfold == fold].reset_index(drop=True)

    # get np arrays to train the model with
    X_train = train.drop(target, axis='columns').values
    y_train = train[target].values

    X_test = test.drop(target, axis='columns').values
    y_test = test[target].values

    # initialise regression tree 
    model = models[model]

    model.fit(X_train, y_train)

    # make predictions and test
    preds = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, preds)

    print(f'Fold = {fold}, MAE = {mae}')

    # joblib.dump(model, f"./dt_{fold}.bin")

if __name__ == '__main__':
    if len(argv) != 3:
        print('Usage: python3 run_fole.py <fold> <model>')
        exit()
    
    if argv[2] not in models.keys():
        print(f'{argv[2]} not found in avaiable models. Available models:')
        print([name for name in models.keys()])
        exit()

    X,y = datasets.make_regression(n_samples=10, n_features=5, n_targets=1)
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    data['target'] = y

    data = init_folds(data, 5)

    run_fold_regression(data=data, fold=int(argv[1]), target='target', model=argv[2])