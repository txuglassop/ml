from sklearn import tree

models = {
    'tree_gini': tree.DecisionTreeRegressor(
        criterion='gini'
    ),
    'tree_entropy': tree.DecisionTreeClassifier(
        criterion='entropy'
    )
}