from asyncio import tasks
from audioop import rms
from tkinter import N
import numpy as np
from combo.models.classifier_comb import SimpleClassifierAggregator
from combo.models.classifier_stacking import Stacking
from combo.models.cluster_comb import ClustererEnsemble, clusterer_ensemble_scores
from combo.models.detector_comb import SimpleDetectorAggregator
from h11 import Data
from numpy import hstack
from sklearn.ensemble import (
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from anai.utils.validator import cv
import pandas as pd


class Ensembler:
    def __init__(
        self,
        task: str,
        n_estimators: int = 3,
        n_clusters: int = 3,
        estimators: list = None,
        verbose: bool = False,
        result_df: pd.DataFrame = None,
        pre_fitted: bool = False,
    ):
        self.n_estimators = n_estimators
        self.n_clusters = n_clusters
        self.estimators = estimators
        self.verbose = verbose
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.task = task
        self.accuracy = None
        self.mae = None
        self.rmse = None
        self.kfold_accuracy = None
        self.pre_fitted = pre_fitted
        self.__ensembling_models_reg = {
            "Stacking Ensembler": self.__stacking_ensembler,
            "Max Voting Ensembler": self.__max_voting,
        }
        self.__ensembling_models_clf = {
            #
            "Stacking Ensembler": self.__stacking_ensembler,
            "Max Voting Ensembler": self.__max_voting,
        }
        self.__ensembling_models_anom = {
            "Cluster Ensembler": self.__cluster_ensembler,
            "Simple Detector Ensembler": self.__simple_detector_agg_ensembler,
        }
        self.result = {}
        self.mode = None

    def ensemble(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        cv_folds: int = 2,
        estimators: list = [],
    ):

        # self.__ensembling_models[self.task]()
        # self.models = models
        cnt = 0
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.X_val = X_val
        self.y_val = y_val
        self.result = pd.DataFrame(index=None)
        mae = []
        rmse = []
        cv_acc = []
        acc = []
        m_list = []
        self.result_anom = {}
        if self.task == "regression":
            self.models_reg = [
                ("lin", LinearRegression()),
                ("sgd", SGDRegressor()),
                ("krr", KernelRidge()),
                ("elas", ElasticNet()),
                ("brr", BayesianRidge()),
            ]
            self.__ensembling_models = self.__ensembling_models_reg
            self.models = self.models_reg if len(estimators) == 0 else estimators
        elif self.task == "classification":
            self.models_clf = [
                ("lr", LogisticRegression()),
                ("sgd", SGDClassifier()),
                ("knn", KNeighborsClassifier()),
                ("svc", SVC()),
            ]
            self.__ensembling_models = self.__ensembling_models_clf
            self.models = self.models_clf if len(estimators) == 0 else estimators
        elif self.task == "anomaly":
            self.__ensembling_models = self.__ensembling_models_anom
            self.models = self.models_clf
        self.result["Name"] = self.__ensembling_models.keys()
        for i, model_name in enumerate(self.__ensembling_models):
            self.model = self.__ensembling_models[model_name]()
            self.evaluator(self.model, model_name)
            acc.append(float("{:.2f}".format(self.accuracy * 100)))
            mae.append(self.mae)
            rmse.append(self.rmse)
            cv_acc.append(float("{:.2f}".format(self.kfold_accuracy.mean() * 100)))
            m_list.append(self.model)
            # self.result[model_name] = [self.accuracy, self.m_absolute_error, self.rm_squared_error, self.model]
        if self.task == "regression":
            self.result["R^2 Score"] = acc
            self.result["Mean Absolute Error"] = mae
            self.result["Root Mean Squared Error"] = rmse
            self.result["Cross Validated Accuracy"] = cv_acc
            self.result["Model"] = m_list
        elif self.task == "classification":
            self.result["Accuracy"] = acc
            self.result["Cross Validated Accuracy"] = cv_acc
            self.result["Model"] = m_list
        elif self.task == "anomaly":
            # self.result_anom[model_name] = [self.accuracy, self.m_absolute_error, self.rm_squared_error, self.model]
            self.result_anom = []
        return self.result if self.task != "anomaly" else self.result_anom

    def __stacking_ensembler(self):
        if self.task == "regression":
            stacking_model = StackingRegressor(
                estimators=self.models,
                final_estimator=LinearRegression(),
                cv=self.cv_folds,
                n_jobs=-1,
                verbose=self.verbose,
            )
        elif self.task == "classification":
            stacking_model = StackingClassifier(
                estimators=self.models,
                final_estimator=LogisticRegression(),
                cv=self.cv_folds,
                n_jobs=-1,
                verbose=self.verbose,
            )
        stacking_model.fit(self.X_train, self.y_train)
        return stacking_model

    def __max_voting(self):
        if self.task == "regression":
            ensembler = VotingRegressor(estimators=self.models)
            ensembler.fit(self.X_train, self.y_train)
        elif self.task == "classification":
            ensembler = VotingClassifier(estimators=self.models)
            ensembler.fit(self.X_train, self.y_train)
        return ensembler

    def __cluster_ensembler(self):
        if self.task == "regression":
            return None
        elif self.task == "classification":
            models = []
            for i in self.models:
                models.append(i[1])
            ensembler_model = ClustererEnsemble(
                models, n_clusters=self.n_clusters, pre_fitted=self.pre_fitted
            )
            ensembler_model.fit(self.X_train)
            return ensembler_model

    def __simple_detector_agg_ensembler(self):
        if self.task == "regression":
            return None
        elif self.task == "classification":
            models = []
            for i in self.models:
                models.append(i[1])
            ensembler_model = SimpleDetectorAggregator(
                base_estimators=models, pre_fitted=self.pre_fitted
            )
            ensembler_model.fit(self.X_train)
            return ensembler_model

    def evaluator(self, model, model_name):
        self.y_pred = model.predict(self.X_val)
        if self.task == "regression":
            self.kfold_accuracy = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv_folds, scoring="r2"
            )
            self.accuracy = r2_score(self.y_val, self.y_pred)
            self.mae = mean_absolute_error(self.y_val, self.y_pred)
            self.rmse = mean_squared_error(self.y_val, self.y_pred, squared=False)
        elif self.task == "classification":
            self.kfold_accuracy = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv_folds, scoring="accuracy"
            )
            self.accuracy = accuracy_score(self.y_val, self.y_pred)

    # def SimpleClassifierAggregator_(self):

    #     clf = SimpleClassifierAggregator(self.models, method='average')
    #     clf.fit(self.X_train, self.y_train)
    #     y_test_predicted = clf.predict(self.X_test)
    #     avg = evaluate_print('Combination by avg   |',
    #                          self.y_test, y_test_predicted)

    #     clf_weights = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
    #     clf = SimpleClassifierAggregator(
    #         self.models, method='average', weights=clf_weights)
    #     clf.fit(self.X_train, self.y_train)
    #     y_test_predicted = clf.predict(self.X_test)
    #     w_avg = evaluate_print('Combination by w_avg |',
    #                            self.y_test, y_test_predicted)

    #     clf = SimpleClassifierAggregator(self.models, method='maximization')
    #     clf.fit(self.X_train, self.y_train)
    #     y_test_predicted = clf.predict(self.X_test)
    #     maximization = evaluate_print(
    #         'Combination by max   |', self.y_test, y_test_predicted)

    #     clf_weights = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
    #     clf = SimpleClassifierAggregator(
    #         self.models, method='majority_vote', weights=clf_weights)
    #     clf.fit(self.X_train, self.y_train)
    #     y_test_predicted = clf.predict(self.X_test)
    #     w_vote = evaluate_print('Combination by w_vote|',
    #                             self.y_test, y_test_predicted)

    #     clf = SimpleClassifierAggregator(self.models, method='median')
    #     clf.fit(self.X_train, self.y_train)
    #     y_test_predicted = clf.predict(self.X_test)
    #     median = evaluate_print('Combination by median|',
    #                             self.y_test, y_test_predicted)

    #     return avg, w_avg, maximization, w_vote, median
    # def fit_ensemble(self, estimators):

    #     meta_X = list()
    #     for name, model in estimators:

    #         model.fit(self.X_train, self.y_train)

    #         yhat = model.predict(self.X_test)

    #         yhat = yhat.view(len(yhat), 1)

    #         meta_X.append(yhat)

    #     meta_X = hstack(meta_X)

    #     blender = LogisticRegression()

    #     blender.fit(meta_X, self.y_test)
    #     return blender

    # def predict_ensemble(estimators, blender, X_test):

    #     meta_X = list()
    #     for name, model in estimators:

    #         yhat = model.predict(X_test)

    #         yhat = yhat.view(len(yhat), 1)

    #         meta_X.append(yhat)

    #     meta_X = hstack(meta_X)

    #     return blender.predict(meta_X)

    # def blending_classifier(self, estimators):

    #     blender = self.fit_ensemble(estimators)
    #     yhat = self.predict_ensemble(estimators, blender, self.X_test)
    #     score = accuracy_score(self.y_test, yhat)
    #     print('Blending Accuracy: %.3f' % (score*100))

    # def fit_ensemble(self, estimators):

    #     meta_X = list()
    #     for name, model in estimators:

    #         model.fit(self.X_train, self.y_train)

    #         yhat = model.predict(self.X_test)

    #         yhat = yhat.view(len(yhat), 1)

    #         meta_X.append(yhat)

    #     meta_X = hstack(meta_X)

    #     blender_r = LinearRegression()

    #     blender_r.fit(meta_X, self.y_test)
    #     return blender_r

    # def predict_ensemble(estimators, blender_r, X_test):

    #     meta_X = list()
    #     for name, model in estimators:

    #         yhat = model.predict(X_test)

    #         yhat = yhat.view(len(yhat), 1)

    #         meta_X.append(yhat)

    #     meta_X = hstack(meta_X)

    #     return blender_r.predict(meta_X)

    # def blending_classifier(self, estimators):
    #     blender_r = self.fit_ensemble(estimators)
    #     yhat = self.predict_ensemble(estimators, blender_r, self.X_test)
