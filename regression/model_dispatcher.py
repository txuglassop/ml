from sklearn import tree, ensemble

models = {
    'tree_sq': tree.DecisionTreeRegressor(
        criterion='squared_error'
    ),
    'tree_pois': tree.DecisionTreeRegressor(
        criterion='poisson'
    ),
    'rf': ensemble.RandomForestRegressor()
}