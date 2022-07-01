intro = """

░█████╗░███╗░░██╗░█████╗░██╗
██╔══██╗████╗░██║██╔══██╗██║
███████║██╔██╗██║███████║██║
██╔══██║██║╚████║██╔══██║██║
██║░░██║██║░╚███║██║░░██║██║
╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝╚═╝
"""


classifiers = {
    "lr": "Logistic Regression",
    "sgd": "Stochastic Gradient Descent",
    "perc": "Perceptron",
    "pass": "Passive Aggressive Classifier",
    "ridg": "Ridge Classifier",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "dt": "Decision Trees",
    "nb": "Naive Bayes",
    "rfc": "Random Forest Classifier",
    "gbc": "Gradient Boosting Classifier",
    "ada": "AdaBoost Classifier",
    "bag": "Bagging Classifier",
    "ext": "Extra Trees Classifier",
    "lgbm": "LightGBM Classifier",
    "cat": "CatBoost Classifier",
    "xgb": "XGBoost Classifier",
    "ann": "Multi Layer Perceptron Classifier",
    "poisson": "Poisson Classifier",
    "huber": "Huber Classifiers",
    "ridge_cv": "RidgeCV Classifier",
    "encv": "ElasticNet CV Classifier",
    "lcv": "LassoCV Classifier",
    "llic": "LassoLarsIC Classifier",
    "llcv": "LassoLarsCV Classifier",
}

regressors = {
    "lin": "Linear Regression",
    "sgd": "Stochastic Gradient Descent Regressor",
    "krr": "Kernel Ridge Regression",
    "elas": "Elastic Net Regression",
    "br": "Bayesian Ridge Regression",
    "svr": "Support Vector Regressor",
    "knn": "K-Nearest Neighbors",
    "dt": "Decision Trees Regressor",
    "rfr": "Random Forest Regressor",
    "gbr": "Gradient Boosted Regressor",
    "ada": "AdaBoostRegressor",
    "bag": "Bagging Regressor",
    "ext": "Extra Trees Regressor",
    "lgbm": "LightGBM Regressor",
    "xgb": "XGBoost Regressor",
    "cat": "Catboost Regressor",
    "ann": "Multi-Layer Perceptron Regressor",
    # "poisson": "Poisson Regressor",
    "huber": "Huber Regressor",
    "gamma": "Gamma Regressor",
    "ridge": "Ridge CV Regressor",
    "encv": "ElasticNetCV Regressor",
    "lcv": "LassoCV Regressor",
    # "llic": "LassoLarsIC Regressor",
    # "llcv": "LassoLarsCV Regressor",
}

classifiers_ver = {
    "lr": "Logistic Regression",
    "sgd": "Stochastic Gradient Descent",
    "perc": "Perceptron",
    "pass": "Passive Aggressive Classifier",
    "ridg": "Ridge Classifier",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "dt": "Decision Trees",
    "nb": "Naive Bayes",
    "rfc": "Random Forest Classifier",
    "gbc": "Gradient Boosting Classifier",
    "ada": "AdaBoost Classifier",
    "bag": "Bagging Classifier",
    "ext": "Extra Trees Classifier",
    "lgbm": "LightGBM Classifier",
    "cat": "CatBoost Classifier",
    "xgb": "XGBoost Classifier",
    "ann": "Multi Layer Perceptron Classifier",
    "poisson": "Poisson Classifier",
    "huber": "Huber Classifiers",
    "ridge_cv": "RidgeCV Classifier",
    "encv": "ElasticNet CV Classifier",
    "lcv": "LassoCV Classifier",
    "llic": "LassoLarsIC Classifier",
    "llcv": "LassoLarsCV Classifier",
    "all": "All Classifiers",
}

regressors_ver = {
    "lin": "Linear Regression",
    "sgd": "Stochastic Gradient Descent Regressor",
    "krr": "Kernel Ridge Regression",
    "elas": "Elastic Net Regression",
    "br": "Bayesian Ridge Regression",
    "svr": "Support Vector Regressor",
    "knn": "K-Nearest Neighbors",
    "dt": "Decision Trees Regressor",
    "rfr": "Random Forest Regressor",
    "gbr": "Gradient Boosted Regressor",
    "ada": "AdaBoostRegressor",
    "bag": "Bagging Regressor",
    "ext": "Extra Trees Regressor",
    "lgbm": "LightGBM Regressor",
    "xgb": "XGBoost Regressor",
    "cat": "Catboost Regressor",
    "ann": "Multi-Layer Perceptron Regressor",
    "poisson": "Poisson Regressor",
    "huber": "Huber Regressor",
    "gamma": "Gamma Regressor",
    "ridge": "Ridge CV Regressor",
    "encv": "ElasticNetCV Regressor",
    "lcv": "LassoCV Regressor",
    "llic": "LassoLarsIC Regressor",
    "llcv": "LassoLarsCV Regressor",
    "all": "All Regressors",
}

params_use_warning = (
    "Params will not work with predictor = 'all'. Settings params = {} "
)
unsupported_pred_warning = """Predictor not available. Please use the predictor which is supported by ANAI.
Check the documentation for more details.\nConflicting Predictor is : {}"""


