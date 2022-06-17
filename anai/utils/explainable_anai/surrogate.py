from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def surrogate_decision_tree(model, X, handler=None, isReg=True):
    new_X = model.predict(X)
    if isReg:
        dt = DecisionTreeRegressor(random_state=42)
    else:
        dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, new_X)
    return dt
