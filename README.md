# ANAI an Automated Machine Learning Library by [Revca](www.revca.io)

![ANAI LOGO](https://revca-assets.s3.ap-south-1.amazonaws.com/Full+version+on+black.jpeg)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/lucifer-ml?period=total&units=international_system&left_color=black&right_color=green&left_text=Total%20Downloads)](https://pepy.tech/project/lucifer-ml)
[![Downloads](https://static.pepy.tech/personalized-badge/lucifer-ml?period=month&units=international_system&left_color=black&right_color=green&left_text=Downloads%20per%20Month)](https://pepy.tech/project/lucifer-ml)
![ReadTheDocs](https://img.shields.io/readthedocs/luciferml?style=plastic)

## About

ANAI is an Automated Machine Learning Python Library that works with tabular data. It is intended to save time when performing data analysis. It will assist you with everything right from the beginning i.e Ingesting data using the inbuilt connectors, preprocessing, feature engineering, model building, model evaluation, model tuning and much more.

## Our Goal

Our Goal is to democratize Machine Learning and make it accessible to everyone.

## Let's get started

### Installation

    1) Python venv:
        pip install anai
    
    2) Anaconda:
        conda install anai

### Available Modelling Techniques

1) Classification

    Available Models for Classification

        - 'lr'  : 'Logistic Regression',
        - 'sgd' : 'Stochastic Gradient Descent',
        - 'perc': 'Perceptron',
        - 'pass': 'Passive Aggressive Classifier',
        - 'ridg': 'Ridge Classifier', 
        - 'svm' : 'Support Vector Machine',
        - 'knn' : 'K-Nearest Neighbours',
        - 'dt'  : 'Decision Trees',
        - 'nb'  : 'Naive Bayes',
        - 'rfc' : 'Random Forest Classifier',
        - 'gbc' : 'Gradient Boosting Classifier',
        - 'ada' : 'AdaBoost Classifier',
        - 'bag' : 'Bagging Classifier',
        - 'extc': 'Extra Trees Classifier',
        - 'lgbm': 'LightGBM Classifier',
        - 'cat' : 'CatBoost Classifier',
        - 'xgb' : 'XGBoost Classifier',
        - 'ann' : 'Multilayer Perceptron Classifier',
        - 'all' : 'Use all the above models',

2) Regression

       Available Models for Regression

        - 'lin' : 'Linear Regression',
        - 'sgd' : 'Stochastic Gradient Descent Regressor',
        - 'elas': 'Elastic Net Regressot',
        - 'krr' : 'Kernel Ridge Regressor',
        - 'br'  : 'Bayesian Ridge Regressor',
        - 'svr' : 'Support Vector Regressor',
        - 'knn' : 'K-Nearest Regressor',
        - 'dt'  : 'Decision Trees',
        - 'rfr' : 'Random Forest Regressor',
        - 'gbr' : 'Gradient Boost Regressor',
        - 'ada' : 'AdaBoost Regressor',
        - 'bag' : 'Bagging Regressor',
        - 'extr': 'Extra Trees Regressor',
        - 'lgbm': 'LightGBM Regressor',
        - 'xgb' : 'XGBoost Regressor',
        - 'cat' : 'Catboost Regressor',
        - 'ann' : 'Multilayer Perceptron Regressor',
        - 'all' : 'Uses all the above models'

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

You can find more examples/tutorials [here](<placeholder>)

## Documentation

More information about ANAI can be found [here](<placeholder>)

## Contributing

- If you have any suggestions or bug reports, please open an issue [here](https://github.com/Revca-ANAI/ANAI/issues)
- If you want to join the ANAI Team send us your resume [here](mailto:careers@revca.io)

## License

- APACHE 2.0 License

## Contact

- [E-mail](mailto:info@anai.io)
- [LinkedIn](https://www.linkedin.com/company/revca/)
- [Website](https://www.anai.io/)

## Roadmap

- [ANAI's roadmap](<placeholder>)
