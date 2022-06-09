from anai.utils.custom_models.wrapped_classifiers import *
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ClassificationObjectives:
    def __init__(
        self, X, y, cv=5, random_state=42, metric="accuracy", lgbm_objective="binary"
    ):
        self.metric = metric
        self.cv = cv
        self.X = X
        self.y = y
        self.random_state = random_state
        self.lgbm_objective = lgbm_objective

    def lr_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "solver": trial.suggest_categorical(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = LogisticRegression(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def sgd_classifier_objective(self, trial):
        param = {
            "loss": trial.suggest_categorical(
                "loss",
                ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            ),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = SGDClassifier(**param)
        scores = cross_val_score(
            clf, self.X, self.y, cv=self.cv, scoring=self.metric, n_jobs=-1
        )
        return scores.mean()

    def ridg_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = RidgeClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def perc_classifier_objective(self, trial):
        param = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = Perceptron(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def pass_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = PassiveAggressiveClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def svm_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "kernel": trial.suggest_categorical(
                "kernel", ["rbf", "linear", "poly", "sigmoid"]
            ),
            "gamma": trial.suggest_loguniform("gamma", 1e-5, 1e5),
            "degree": trial.suggest_int("degree", 1, 10),
            "coef0": trial.suggest_loguniform("coef0", 1e-5, 1e5),
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = SVC(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def knn_classifier_objective(self, trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 256),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 10),
            "n_jobs": -1,
            "rows_limit": 100000,
        }
        clf = KNeighborsClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def dt_classifier_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "random_state": self.random_state,
        }
        clf = DecisionTreeClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def rfc_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "max_features": trial.suggest_float("max_features", 0.01, 1),
            "seed": self.random_state,
            "n_jobs": -1,
            "max_steps": 10,
        }
        clf = RandomForestClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def gbc_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
        clf = GradientBoostingClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def ada_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
            "random_state": self.random_state,
        }
        clf = AdaBoostClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def bag_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "bootstrap_features": trial.suggest_categorical(
                "bootstrap_features", [True, False]
            ),
            "max_samples": trial.suggest_uniform("max_samples", 0.1, 1),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1),
            "n_jobs": -1,
            "random_state": self.random_state,
        }
        clf = BaggingClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def extc_classifier_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "max_features": trial.suggest_float("max_features", 0.01, 1),
            "random_state": self.random_state,
            "n_jobs": -1,
            "max_steps": 10,
        }
        clf = ExtraTreesClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def lgbm_classifier_objective(self, trial):
        param = {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.0125, 0.025, 0.05, 0.1]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": min(
                trial.suggest_float("feature_fraction", 0.3, 1.0 + 1e-8), 1.0
            ),
            "bagging_fraction": min(
                trial.suggest_float("bagging_fraction", 0.3, 1.0 + 1e-8), 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "feature_pre_filter": False,
            "boosting_type": "gbdt",
            "seed": self.random_state,
            "num_threads": -1,
            "objective": self.lgbm_objective,
        }
        clf = LGBMClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def cat_classifier_objective(self, trial):
        param = {
            "objective": trial.suggest_categorical(
                "objective", ["Logloss", "CrossEntropy"]
            ),
            "iterations": 1000,
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.05, 0.1, 0.2]
            ),
            "rsm": trial.suggest_float("rsm", 0.1, 1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0001, 10.0, log=False),
            "random_state": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        clf = CatBoostClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def xgb_classifier_objective(self, trial):
        param = {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.05, 0.1, 0.2]
            ),
            "eta": trial.suggest_categorical("eta", [0.0125, 0.025, 0.05, 0.1]),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "colsample_bytree": min(
                trial.suggest_float("colsample_bytree", 0.3, 1.0 + 1e-8), 1.0
            ),
            "subsample": min(trial.suggest_float("subsample", 0.3, 1.0 + 1e-8), 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "tree_method": "hist",
            "booster": "gbtree",
            "n_jobs": -1,
            "seed": self.random_state,
            "verbosity": 0,
        }
        clf = XGBClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def mlp_classifier_objective(self, trial):
        param = {
            "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 1, 10),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "dense_1_size": trial.suggest_int("dense_1_size", 4, 100),
            "dense_2_size": trial.suggest_int("dense_2_size", 2, 100),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.005, 0.01, 0.05, 0.1, 0.2]
            ),
            "learning_rate_type": trial.suggest_categorical(
                "learning_rate_type", ["constant", "adaptive"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "seed": self.random_state,
        }
        clf = MLPClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def poisson_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
        }
        classifier = PoissonClassifier(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def huber_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "epsilon": trial.suggest_loguniform("epsilon", 1e-8, 10.0),
        }
        param["loss"] = "modified_huber"
        classifier = SGDClassifier(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def gamma_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
        }
        classifier = GammaClassifier(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def ridge_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "random_state": self.random_state,
            "normalize": False,
        }
        classifier = RidgeClassifier(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def encv_classifier_objective(self, trial):
        param["penalty"] = "elasticnet"
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "l1_ratio": trial.suggest_loguniform("l1_ratio", 1e-8, 1e-1),
            "eps": trial.suggest_loguniform("epsilon", 1e-8, 1e-1),
            "n_alphas": trial.suggest_int("n_alphas", 1, 2000),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        classifier = LogisticRegressionCV(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def lcv_classifier_objective(self, trial):
        param["penalty"] = "l1"
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "eps": trial.suggest_loguniform("epsilon", 1e-8, 1e-1),
            "n_alphas": trial.suggest_int("n_alphas", 1, 2000),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        classifier = LogisticRegressionCV(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def llic_classifier_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical("criterion", ["aic", "bic"]),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "normalize": False,
        }
        classifier = LassoLarsIC(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def llcv_classifier_objective(self, trial):
        param = {
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "max_n_alphas": trial.suggest_int("max_n_alphas", 1, 2000),
            "eps": trial.suggest_loguniform("epsilon", 1e-8, 1e-1),
            "normalize": False,
            "n_jobs": -1,
        }
        classifier = LassoLarsCV(**param)
        scores = cross_val_score(
            classifier, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()
