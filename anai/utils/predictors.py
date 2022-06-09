import traceback

from anai.utils.custom_models.wrapped_classifiers import *
from anai.utils.tuner.optuna.objectives.classification_objectives import \
    ClassificationObjectives
from anai.utils.tuner.optuna.objectives.regression_objectives import \
    RegressionObjectives
from catboost import CatBoostClassifier, CatBoostRegressor
from colorama import Fore
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, ElasticNetCV,
                                  GammaRegressor, HuberRegressor, LassoCV,
                                  LassoLarsCV, LassoLarsIC, LinearRegression,
                                  LogisticRegression, LogisticRegressionCV,
                                  PassiveAggressiveClassifier, Perceptron,
                                  PoissonRegressor, Ridge, RidgeClassifier,
                                  SGDClassifier, SGDRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def classification_predictor(
    predictor,
    params,
    X_train,
    y_train,
    cv_folds,
    random_state,
    metric,
    mode="single",
    verbose=False,
    lgbm_objective="binary",
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a classifier for the Choosen Predictor.
    """
    try:
        objective_to_be_tuned = None
        objective = ClassificationObjectives(
            X_train,
            y_train,
            cv=cv_folds,
            random_state=random_state,
            metric=metric,
            lgbm_objective=lgbm_objective,
        )
        if predictor == "lr":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training Logistic Regression on Training Set [*]\n"
                )
            classifier = LogisticRegression(**params)
            objective_to_be_tuned = objective.lr_classifier_objective
        elif predictor == "sgd":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Stochastic Gradient Descent on Training Set [*]\n"
                )
            classifier = SGDClassifier(**params)
            objective_to_be_tuned = objective.sgd_classifier_objective
        elif predictor == "perc":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Perceptron on Training Set [*]\n")
            classifier = Perceptron(**params)
            objective_to_be_tuned = objective.perc_classifier_objective
        elif predictor == "pass":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Passive Aggressive on Training Set [*]\n")
            classifier = PassiveAggressiveClassifier(**params)
            objective_to_be_tuned = objective.pass_classifier_objective
        elif predictor == "ridg":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Ridge Classifier on Training Set [*]\n")
            classifier = RidgeClassifier(**params)
            objective_to_be_tuned = objective.ridg_classifier_objective
        elif predictor == "svm":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Support Vector Machine on Training Set [*]\n"
                )
            classifier = SVC(**params)
            objective_to_be_tuned = objective.svm_classifier_objective
        elif predictor == "knn":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training K-Nearest Neighbors on Training Set [*]\n"
                )
            classifier = KNeighborsClassifier(**params)
            objective_to_be_tuned = objective.knn_classifier_objective
        elif predictor == "dt":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Decision Tree Classifier on Training Set [*]\n"
                )
            classifier = DecisionTreeClassifier(**params)
            objective_to_be_tuned = objective.dt_classifier_objective
        elif predictor == "nb":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Naive Bayes Classifier on Training Set [*]\n"
                )
            classifier = GaussianNB(**params)
            objective_to_be_tuned = None
        elif predictor == "rfc":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Random Forest Classifier on Training Set [*]\n"
                )
            classifier = RandomForestClassifier(**params)
            objective_to_be_tuned = objective.rfc_classifier_objective
        elif predictor == "gbc":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Gradient Boosting Classifier on Training Set [*]\n"
                )
            classifier = GradientBoostingClassifier(**params)
            objective_to_be_tuned = objective.gbc_classifier_objective
        elif predictor == "ada":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training AdaBoost Classifier on Training Set [*]\n"
                )
            classifier = AdaBoostClassifier(**params)
            objective_to_be_tuned = objective.ada_classifier_objective
        elif predictor == "bag":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Bagging Classifier on Training Set [*]\n")
            classifier = BaggingClassifier(**params)
            objective_to_be_tuned = objective.bag_classifier_objective
        elif predictor == "ext":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Extra Trees Classifier on Training Set [*]\n"
                )
            classifier = ExtraTreesClassifier(**params)
            objective_to_be_tuned = objective.extc_classifier_objective
        elif predictor == "lgbm":
            if mode == "single":
                print(Fore.YELLOW + "Training LightGBM on Training Set [*]\n")
            classifier = LGBMClassifier(**params)
            objective_to_be_tuned = objective.lgbm_classifier_objective
        elif predictor == "cat":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training CatBoostClassifier on Training Set [*]\n")
            params["verbose"] = verbose
            classifier = CatBoostClassifier(**params)
            params.pop("verbose")
            objective_to_be_tuned = objective.cat_classifier_objective
        elif predictor == "xgb":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training XGBClassifier on Training Set [*]\n")
            if verbose:
                params["verbosity"] = 2
            if not verbose:
                params["verbosity"] = 0
            params["use_label_encoder"] = False
            params["eval_metric"] = (
                params["eval_metric"] if "eval_metric" in params else "logloss"
            )
            classifier = XGBClassifier(**params)
            objective_to_be_tuned = objective.xgb_classifier_objective
            params.pop("verbosity")
            params.pop("eval_metric")
            params.pop("use_label_encoder")
        elif predictor == "ann":
            classifier = MLPClassifier(**params)
            objective_to_be_tuned = objective.mlp_classifier_objective
        elif predictor == "poisson":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Poisson Classifier on Training Set [*]\n")
            classifier = PoissonClassifier(**params)
            objective_to_be_tuned = objective.poisson_classifier_objective
        elif predictor == "huber":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Huber Classifier on Training Set [*]\n")
            params["loss"] = "modified_huber"
            classifier = SGDClassifier()
            objective_to_be_tuned = objective.huber_classifier_objective
            params.pop("loss")
        elif predictor == "ridge_cv":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training RidgeCV Classifier on Training Set [*]\n")
            classifier = LogisticRegressionCV()
            objective_to_be_tuned = objective.ridg_classifier_objective
        elif predictor == "encv":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training ElasticnetCV Classifier on Training Set [*]\n"
                )
            params["penalty"] = "elasticnet"
            classifier = LogisticRegressionCV()
            objective_to_be_tuned = objective.encv_classifier_objective
            params.pop("penalty")
        elif predictor == "lcv":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training LassoCV Classifier on Training Set [*]\n")
            params["penalty"] = "l1"
            classifier = LogisticRegressionCV()
            objective_to_be_tuned = objective.lcv_classifier_objective
            params.pop("penalty")
        elif predictor == "llic":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Lasso Lars IC on Training Set [*]\n")
            params["normalize"] = False
            classifier = LassoLarsICClassifier(**params)
            objective_to_be_tuned = objective.llic_classifier_objective
            params.pop("normalize")
        elif predictor == "llcv":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Lasso Lars CV on Training Set [*]\n")
            params["normalize"] = False
            classifier = LassoLarsCVClassifier(**params)
            objective_to_be_tuned = objective.llcv_classifier_objective
            params.pop("normalize")
        return (classifier, objective_to_be_tuned)
    except Exception as error:
        print(traceback.format_exc())
        print(Fore.RED + "Model Build Failed with error :", error, "\n")


def regression_predictor(
    predictor,
    params,
    X_train,
    y_train,
    cv_folds,
    random_state,
    metric,
    mode="single",
    verbose=False,
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a regressor for the Chosen Predictor.
    """
    try:
        objective = RegressionObjectives(
            X_train, y_train, cv=cv_folds, random_state=random_state, metric=metric
        )
        if predictor == "lin":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Linear Regression on Training Set [*]\n")
            regressor = LinearRegression(**params)
            objective_to_be_tuned = objective.lin_regressor_objective
        elif predictor == "sgd":
            if mode == "single":
                print(
                    "Training Stochastic Gradient Descent Regressor on Training Set [*]\n"
                )
            regressor = SGDRegressor(**params)
            objective_to_be_tuned = objective.sgd_regressor_objective
        elif predictor == "krr":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Kernel Ridge Regressor on Training Set [*]\n"
                )
            regressor = KernelRidge(**params)
            objective_to_be_tuned = objective.krr_regressor_objective
        elif predictor == "elas":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training ElasticNet Regressor on Training Set [*]\n"
                )
            regressor = ElasticNet(**params)
            objective_to_be_tuned = objective.elas_regressor_objective
        elif predictor == "br":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training BayesianRidge Regressor on Training Set [*]\n"
                )
            regressor = BayesianRidge(**params)
            objective_to_be_tuned = objective.br_regressor_objective
        elif predictor == "svr":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Support Vector Machine on Training Set [*]\n"
                )
            regressor = SVR(**params)
            objective_to_be_tuned = objective.svr_regressor_objective
        elif predictor == "knn":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training KNeighbors Regressor on Training Set [*]\n"
                )
            params["n_jobs"] = -1
            regressor = KNeighborsRegressor(**params)
            objective_to_be_tuned = objective.knr_regressor_objective
            params.pop("n_jobs")
        elif predictor == "dt":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Decision Tree regressor on Training Set [*]\n"
                )
            regressor = DecisionTreeRegressor(**params)
            objective_to_be_tuned = objective.dt_regressor_objective
        elif predictor == "rfr":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Random Forest regressor on Training Set [*]\n"
                )
            params["n_jobs"] = -1
            params["random_state"] = random_state
            regressor = RandomForestRegressor(**params)
            objective_to_be_tuned = objective.rfr_regressor_objective
            params.pop("n_jobs")
            params.pop("random_state")
        elif predictor == "gbr":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Gradient Boosting Regressor  on Training Set [*]\n"
                )
            regressor = GradientBoostingRegressor(**params)
            objective_to_be_tuned = objective.gbr_regressor_objective
        elif predictor == "ada":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training AdaBoost Regressor on Training Set [*]\n")
            regressor = AdaBoostRegressor(**params)
            objective_to_be_tuned = objective.ada_regressor_objective
        elif predictor == "bag":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Bagging Regressor on Training Set [*]\n")
            params["n_jobs"] = -1
            regressor = BaggingRegressor(**params)
            objective_to_be_tuned = objective.bag_regressor_objective
            params.pop("n_jobs")
        elif predictor == "ext":
            if mode == "single":
                print(
                    Fore.YELLOW +
                    "Training Extra Trees Regressor on Training Set [*]\n"
                )
            params["n_jobs"] = -1
            regressor = ExtraTreesRegressor(**params)
            objective_to_be_tuned = objective.extr_regressor_objective
            params.pop("n_jobs")
        elif predictor == "xgb":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training XGBregressor on Training Set [*]\n")
            regressor = XGBRegressor(**params)
            objective_to_be_tuned = objective.xgb_regressor_objective
        elif predictor == "lgbm":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training LGBMRegressor on Training Set [*]\n")
            regressor = LGBMRegressor(**params)
            objective_to_be_tuned = objective.lgbm_regressor_objective
        elif predictor == "cat":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training CatBoost Regressor on Training Set [*]\n")
            params["thread_count"] = -1
            params["verbose"] = verbose
            regressor = CatBoostRegressor(**params)
            params.pop("verbose")
            objective_to_be_tuned = objective.cat_regressor_objective
            params.pop("thread_count")
        elif predictor == "ann":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Multi Layered Perceptron on Training Set [*]\n"
                )
            params["max_iter"] = (
                2000 if "max_iter" not in params.keys() else params["max_iter"]
            )
            regressor = MLPRegressor(**params)
            objective_to_be_tuned = objective.mlp_regressor_objective
            params.pop("max_iter")
        elif predictor == "poisson":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Generalized Linear Model with a Poisson distribution on Training Set [*]\n"
                )
            params["verbose"] = verbose
            regressor = PoissonRegressor(**params)
            objective_to_be_tuned = objective.poisson_regressor_objective
            params.pop("verbose")
        elif predictor == "huber":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Huber Regressor on Training Set [*]\n")
            regressor = HuberRegressor(**params)
            objective_to_be_tuned = objective.huber_regressor_objective
        elif predictor == "gamma":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Generalized Linear Model with a Gamma distribution on Training Set [*]\n"
                )
            params["verbose"] = verbose
            regressor = GammaRegressor(**params)
            objective_to_be_tuned = objective.gamma_regressor_objective
            params.pop("verbose")
        elif predictor == "ridge":
            if mode == "single":
                print(
                    Fore.YELLOW
                    + "Training Linear least squares with l2 regularization on Training Set [*]\n"
                )
            regressor = Ridge(**params)
            objective_to_be_tuned = objective.ridge_regressor_objective
        elif predictor == "encv":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training Elastic Net CV on Training Set [*]\n")
            params["verbose"] = verbose
            params["n_jobs"] = -1
            regressor = ElasticNetCV(**params)
            objective_to_be_tuned = objective.encv_regressor_objective
            params.pop("verbose")
            params.pop("n_jobs")
        elif predictor == "lcv":
            if mode == "single":
                print(Fore.YELLOW + "Training Lasso CV on Training Set [*]\n")
            params["verbose"] = verbose
            params["n_jobs"] = -1
            regressor = LassoCV(**params)
            objective_to_be_tuned = objective.lcv_regressor_objective
            params.pop("verbose")
            params.pop("n_jobs")
        elif predictor == "llic":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training LassoLarsIC on Training Set [*]\n")
            params["verbose"] = verbose
            regressor = LassoLarsIC(**params)
            objective_to_be_tuned = objective.llic_regressor_objective
            params.pop("verbose")
        elif predictor == "llcv":
            if mode == "single":
                print(Fore.YELLOW +
                      "Training LassoLarsCV on Training Set [*]\n")
            params["verbose"] = verbose
            params["n_jobs"] = -1
            regressor = LassoLarsCV(**params)
            objective_to_be_tuned = objective.llcv_regressor_objective
            params.pop("verbose")
            params.pop("n_jobs")
        return (regressor, objective_to_be_tuned)
    except Exception as error:
        print(traceback.format_exc())
        params = None
        print(Fore.RED + "Model Build Failed with error :", error, "\n")
