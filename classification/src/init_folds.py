from sklearn.model_selection import KFold
from sklearn import datasets

def init_folds(data, num_folds=5):
    """
    function that adds a new column 'kfold' which is an integer 0 <= k < num_folds

    Parameters:
        :param data - pandas df to add new column to
        :param num_folds - number of folds to add data to 

    returns:
        same df with new column added
    """
    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # initiate kf class
    kf = KFold(n_splits=num_folds)

    data['kfold'] = -1

    for fold, (trn, val) in enumerate(kf.split(X=data)):
        data.loc[val, 'kfold'] = fold

    return data
