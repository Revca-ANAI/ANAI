from re import S

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    GammaRegressor,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PoissonRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class RegressionObjectives:
    def __init__(self, X, y, cv=5, random_state=42, metric="R^2"):
        self.metric = metric
        self.cv = cv
        self.X = X
        self.y = y
        self.random_state = random_state

    def lin_regressor_objective(self, trial):
        param = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "copy_X": trial.suggest_categorical("copy_X", [True, False]),
        }
        regressor = LinearRegression(**param, n_jobs=-1)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, scoring=self.metric
        )
        return scores.mean()

    def sgd_regressor_objective(self, trial):
        param = {
            "loss": trial.suggest_categorical(
                "loss", ["squared_loss", "huber", "epsilon_insensitive"]
            ),
            "penalty": trial.suggest_categorical(
                "penalty", ["none", "l2", "l1", "elasticnet"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-3),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
            ),
            "eta0": trial.suggest_float("eta0", 0.0, 1.0),
            "power_t": trial.suggest_float("power_t", 0.0, 1.0),
            "warm_start": trial.suggest_categorical("warm_start", [True, False]),
            "average": trial.suggest_categorical("average", [True, False]),
            "random_state": self.random_state,
        }
        regressor = SGDRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def krr_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-10, 1e-3),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "degree": trial.suggest_int("degree", 1, 3),
            "gamma": trial.suggest_loguniform("gamma", 1e-10, 1e-3),
            "coef0": trial.suggest_loguniform("coef0", 1e-10, 1e-3),
        }
        regressor = KernelRidge(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def elas_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-10, 1e-3),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
            "tol": trial.suggest_loguniform("tol", 1e-10, 1e-3),
            "random_state": self.random_state,
        }
        regressor = ElasticNet(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def br_regressor_objective(self, trial):
        param = {
            "alpha_1": trial.suggest_loguniform("alpha_1", 1e-10, 1e-3),
            "alpha_2": trial.suggest_loguniform("alpha_2", 1e-10, 1e-3),
            "lambda_1": trial.suggest_loguniform("lambda_1", 1e-10, 1e-3),
            "lambda_2": trial.suggest_loguniform("lambda_2", 1e-10, 1e-3),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
            "copy_X": trial.suggest_categorical("copy_X", [True, False]),
        }
        regressor = BayesianRidge(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def svr_regressor_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-10, 1e-3),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "degree": trial.suggest_int("degree", 1, 3),
            "gamma": trial.suggest_loguniform("gamma", 1e-10, 1e-3),
            "coef0": trial.suggest_loguniform("coef0", 1e-10, 1e-3),
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "tol": trial.suggest_loguniform("tol", 1e-10, 1e-3),
            "cache_size": trial.suggest_loguniform("cache_size", 1e-10, 1e-3),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
        regressor = SVR(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def knr_regressor_objective(self, trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 1, 100),
            "p": trial.suggest_int("p", 1, 3),
            "n_jobs": -1,
        }
        regressor = KNeighborsRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def dt_regressor_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical(
                "criterion", ["mse", "friedman_mse", "mae"]
            ),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_loguniform(
                "min_weight_fraction_leaf", 1e-10, 1e-3
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2", None]
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 10),
            "min_impurity_decrease": trial.suggest_loguniform(
                "min_impurity_decrease", 1e-10, 1e-3
            ),
            "random_state": self.random_state,
        }
        regressor = DecisionTreeRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def gbr_regressor_objective(self, trial):
        param = {
            "loss": trial.suggest_categorical(
                "loss", ["ls", "lad", "huber", "quantile"]
            ),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-10, 1e-3),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "criterion": trial.suggest_categorical(
                "criterion", ["friedman_mse", "mae"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_loguniform(
                "min_weight_fraction_leaf", 1e-10, 1e-3
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2", None]
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 10),
            "min_impurity_decrease": trial.suggest_loguniform(
                "min_impurity_decrease", 1e-10, 1e-3
            ),
            "random_state": self.random_state,
        }
        regressor = GradientBoostingRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def ada_regressor_objective(self, trial):
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-10, 1e-3),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "loss": trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            ),
            "random_state": self.random_state,
        }
        regressor = AdaBoostRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def bag_regressor_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "bootstrap_features": trial.suggest_categorical(
                "bootstrap_features", [True, False]
            ),
            "oob_score": trial.suggest_categorical("oob_score", [True, False]),
            "max_samples": trial.suggest_uniform("max_samples", 0.0, 1.0),
            "max_features": trial.suggest_uniform("max_features", 0.0, 1.0),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        regressor = BaggingRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def extr_regressor_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "criterion": trial.suggest_categorical(
                "criterion", ["mse", "friedman_mse", "mae"]
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_loguniform(
                "min_weight_fraction_leaf", 1e-10, 1e-3
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2", None]
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 10),
            "min_impurity_decrease": trial.suggest_loguniform(
                "min_impurity_decrease", 1e-10, 1e-3
            ),
            "bootstrap": True,
            "oob_score": trial.suggest_categorical("oob_score", [True, False]),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        regressor = ExtraTreesRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def rfr_regressor_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
            "max_depth": trial.suggest_int("max_depth", 10, 80, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 9),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": 0,
        }
        regressor = RandomForestRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def xgb_regressor_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
            "max_depth": trial.suggest_int("max_depth", 8, 16),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
            "gamma": trial.suggest_int("gamma", 1, 3),
            "learning_rate": 0.01,
            "colsample_bytree": trial.suggest_discrete_uniform(
                "colsample_bytree", 0.5, 1, 0.1
            ),
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
            "random_state": 42,
        }
        regressor = XGBRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def cat_regressor_objective(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 4, 10),
            "random_strength": trial.suggest_int("random_strength", 0, 100),
            "bagging_temperature": trial.suggest_loguniform(
                "bagging_temperature", 0.01, 100.00
            ),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        }
        regressor = CatBoostRegressor(**params)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def lgbm_regressor_objective(self, trial):

        param = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.0125, 0.025, 0.05, 0.1]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "colsample_bytree": min(
                trial.suggest_float("colsample_bytree", 0.3, 1.0 + 1e-8), 1.0
            ),
            "bagging_fraction": min(
                trial.suggest_float("bagging_fraction", 0.3, 1.0 + 1e-8), 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_pre_filter": False,
            "random_state": self.random_state,
            "num_threads": -1,
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        regressor = LGBMRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def mlp_regressor_objective(self, trial):
        param = {
            "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 2, 10),
            "activation": trial.suggest_categorical("activation", ["logistic", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "adam"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "adaptive"]
            ),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "random_state": self.random_state,
            "verbose": 0,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 10,
        }
        regressor = MLPRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def poisson_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
        }
        regressor = PoissonRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def huber_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "epsilon": trial.suggest_loguniform("epsilon", 1e-8, 10.0),
        }
        regressor = HuberRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def gamma_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
        }
        regressor = GammaRegressor(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def ridge_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "random_state": self.random_state,
            "normalize": False,
        }
        regressor = Ridge(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def encv_regressor_objective(self, trial):
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
        regressor = ElasticNetCV(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def lcv_regressor_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            "eps": trial.suggest_loguniform("epsilon", 1e-8, 1e-1),
            "n_alphas": trial.suggest_int("n_alphas", 1, 2000),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "tol": trial.suggest_loguniform("tol", 1e-8, 1e-1),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        regressor = LassoCV(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def llic_regressor_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical("criterion", ["aic", "bic"]),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "normalize": False,
        }
        regressor = LassoLarsIC(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()

    def llcv_regressor_objective(self, trial):
        param = {
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "max_n_alphas": trial.suggest_int("max_n_alphas", 1, 2000),
            "eps": trial.suggest_loguniform("epsilon", 1e-8, 1e-1),
            "normalize": False,
            "n_jobs": -1,
        }
        regressor = LassoLarsCV(**param)
        scores = cross_val_score(
            regressor, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.metric
        )
        return scores.mean()
