import pandas as pd

from init_folds import init_folds
from run_fold import run_fold_logistic, run_fold_rf, run_fold_xgb

if __name__ == '__main__':
    data = pd.read_csv('../input/train.csv')
    data = init_folds(data)

    for fold in range(5):
        run_fold_xgb(data, fold)
