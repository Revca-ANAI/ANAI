# ANAI an Automated Machine Learning Library by [Revca](https://revca.io/)

![ANAI LOGO](https://revca-assets.s3.ap-south-1.amazonaws.com/Blue+Yellow+Futuristic+Virtual+Technology+Blog+Banner.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/lucifer-ml?period=total&units=international_system&left_color=black&right_color=green&left_text=Total%20Downloads)](https://pepy.tech/project/lucifer-ml)
[![Downloads](https://static.pepy.tech/personalized-badge/lucifer-ml?period=month&units=international_system&left_color=black&right_color=green&left_text=Downloads%20per%20Month)](https://pepy.tech/project/lucifer-ml)

## About

ANAI is an Automated Machine Learning Python Library that works with tabular data. It is intended to save time when performing data analysis. It will assist you with everything right from the beginning i.e Ingesting data using the inbuilt connectors, preprocessing, feature engineering, model building, model evaluation, model tuning and much more.

## Our Goal

Our Goal is to democratize Machine Learning and make it accessible to everyone.

## Let's get started

### Installation

    1) Python venv:
        pip install anai-opensource

### Available Modelling Techniques

1) Classification

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

2) Regression

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

### Usage Example

    import anai
    ai = anai.run(
                filepath='examples/Folds5x2_pp.xlsx', 
                target='PE',
                predictor=['lin'],
    )

![ANAI Run](https://revca-assets.s3.ap-south-1.amazonaws.com/anai_run_example.png)

### Hyperparameter Tuning

ANAI is powered by [Optuna](https://github.com/optuna/optuna) for Hyperparam tuning. Just pass "tune = True" in run arguments and it will start tuning the model/s with Optuna.

### Persistence

ANAI's model can be saved as a pickle file. It will save both the model and the scaler to the pickle file.

    - Saving

        Ex: 
            ai.save([<path-to-model.pkl>, <path-to-scaler.pkl>])

A new ANAI Object can be loaded as well by specifying path of model and scaler

    - Loading

        Ex: 
            ai = anai.run(path = [<path-to-model.pkl>, <path-to-scaler.pkl>])

## More Examples

You can find more examples/tutorials [here](https://github.com/Revca-ANAI/ANAI/examples)

## Documentation

More information about ANAI can be found [here](https://github.com/Revca-ANAI/ANAI/tree/main/docs)

## Contributing

- If you have any suggestions or bug reports, please open an issue [here](https://github.com/Revca-ANAI/ANAI/issues)
- If you want to join the ANAI Team send us your resume [here](mailto:careers@revca.io)

## License

- APACHE 2.0 License

## Contact

- [E-mail](mailto:info@anai.io)
- [LinkedIn](https://www.linkedin.com/company/revca-io/)
- [Website](https://www.anai.io/)

## Roadmap

- [ANAI's roadmap](https://anai.io/)

