# ANAI

## Available Algorithms

### Regression

    Available Models for Regression

    - "lin": "Linear Regression"
    - "sgd": "Stochastic Gradient Descent Regressor"
    - "krr": "Kernel Ridge Regression"
    - "elas": "Elastic Net Regression"
    - "br": "Bayesian Ridge Regression"
    - "svr": "Support Vector Regressor"
    - "knn": "K-Nearest Neighbors"
    - "dt": "Decision Trees Regressor"
    - "rfr": "Random Forest Regressor"
    - "gbr": "Gradient Boosted Regressor"
    - "ada": "AdaBoostRegressor"
    - "bag": "Bagging Regressor"
    - "ext": "Extra Trees Regressor"
    - "lgbm": "LightGBM Regressor"
    - "xgb": "XGBoost Regressor"
    - "cat": "Catboost Regressor"
    - "ann": "Multi-Layer Perceptron Regressor"
    - "poisson": "Poisson Regressor"
    - "huber": "Huber Regressor"
    - "gamma": "Gamma Regressor"
    - "ridge": "Ridge CV Regressor"
    - "encv": "ElasticNetCV Regressor"
    - "lcv": "LassoCV Regressor"
    - "llic": "LassoLarsIC Regressor"
    - "llcv": "LassoLarsCV Regressor"
    - "all": "All Regressors"

### Classification

       Available Models for Classification

        - "lr": "Logistic Regression"
        - "sgd": "Stochastic Gradient Descent"
        - "perc": "Perceptron"
        - "pass": "Passive Aggressive Classifier"
        - "ridg": "Ridge Classifier"
        - "svm": "Support Vector Machine"
        - "knn": "K-Nearest Neighbors"
        - "dt": "Decision Trees"
        - "nb": "Naive Bayes"
        - "rfc": "Random Forest Classifier"
        - "gbc": "Gradient Boosting Classifier"
        - "ada": "AdaBoost Classifier"
        - "bag": "Bagging Classifier"
        - "ext": "Extra Trees Classifier"
        - "lgbm": "LightGBM Classifier"
        - "cat": "CatBoost Classifier"
        - "xgb": "XGBoost Classifier"
        - "ann": "Multi Layer Perceptron Classifier"
        - "poisson": "Poisson Classifier"
        - "huber": "Huber Classifiers"
        - "ridge_cv": "RidgeCV Classifier"
        - "encv": "ElasticNet CV Classifier"
        - "lcv": "LassoCV Classifier"
        - "llic": "LassoLarsIC Classifier"
        - "llcv": "LassoLarsCV Classifier"
        - "all": "All Classifiers"

### Model Explanation

    Available Explanation Methods
        - 'perm': Permutation Importance
        - 'shap': Shapley Importance

### Anomaly Detection

ANAI uses [PYOD](https://github.com/yzhao062/pyod) for anomaly detection.

    Available Models for Anomaly Detection  
        - "IForest": "Isolation Forest"
        - "CBLOF": "Cluster-based Local Outlier Factor"

### Missing Data handling

ANAI supports statistical imputing

    Available Methods for Missing Data Handling
        - "mean": "Mean Imputation"
        - "median": "Median Imputation"
        - "mode": "Mode Imputation"
        - "drop": "Drop Missing Data"

### Categorical Encoding

ANAI uses catboost encoder for categorical feature encoding and Label Encoding for categorical target

### Scaling

ANAI supports both standard and normalization scaling.

    Available Scaling Methods
        - "standardize": "Standard Scaling"
        - "normal": "Normalization Scaling"
