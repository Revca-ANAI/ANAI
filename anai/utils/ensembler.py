import numpy as np
import pandas as pd

from mlxtend.classifier import (EnsembleVoteClassifier, StackingClassifier,
                                StackingCVClassifier)
from mlxtend.regressor import StackingCVRegressor, StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, LinearRegression,
                                  LogisticRegression, SGDClassifier,
                                  SGDRegressor)
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Ensembler:
    def __init__(self, task: str, n_estimators: int = 3, n_clusters: int = 3, estimators: list = None, verbose: bool = False, random_state: int = 42, pre_fitted: bool = False):
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
        self.random_state = random_state
        self.est = None
        self.__ensembling_models_reg = {
            'Stacking Ensembler': self._stacking_ensembler,
            # 'Stacking Ensembler with CV': self._stacking_ensembler_cv,
            'Max Voting Ensembler': self._max_voting
        }
        self.__ensembling_models_clf = {
            #
            'Stacking Ensembler': self._stacking_ensembler,
            'Stacking Ensembler with CV': self._stacking_ensembler_cv,
            'Max Voting Ensembler': self._max_voting
        }
        self.result = {}
        self.mode = None

    def ensemble(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, cv_folds: int = 2, estimators: list = [], est: list = None):

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
        self.est = est
        if self.task == 'regression':
            self.models_reg = [('lin', LinearRegression()), ('sgd', SGDRegressor(
            )), ('krr', KernelRidge()), ('elas', ElasticNet()), ('brr', BayesianRidge())]
            self.__ensembling_models = self.__ensembling_models_reg
            self.models = self.models_reg if len(
                estimators) == 0 else estimators
        elif self.task == 'classification':
            self.models_clf = [('lr', LogisticRegression()), ('sgd', SGDClassifier(
            )), ('knn', KNeighborsClassifier()), ('svc', SVC())]
            self.__ensembling_models = self.__ensembling_models_clf
            self.models = self.models_clf if len(
                estimators) == 0 else estimators
        elif self.task == 'anomaly':
            self.__ensembling_models = self.__ensembling_models_anom
            self.models = self.models_clf
        self.result['Name'] = self.__ensembling_models.keys()
        for i, model_name in enumerate(self.__ensembling_models):
            self.model = self.__ensembling_models[model_name]()
            self.evaluator(self.model, model_name)
            acc.append(float('{:.2f}'.format(self.accuracy*100)))
            mae.append(self.mae)
            rmse.append(self.rmse)
            cv_acc.append(float('{:.2f}'.format(
                self.kfold_accuracy.mean()*100)))
            m_list.append(self.model)
            # self.result[model_name] = [self.accuracy, self.m_absolute_error, self.rm_squared_error, self.model]
        if self.task == 'regression':
            self.result["R^2 Score"] = acc
            self.result["Mean Absolute Error"] = mae
            self.result["Root Mean Squared Error"] = rmse
            self.result["Cross Validated Accuracy"] = cv_acc
            self.result["Model"] = m_list
        elif self.task == 'classification':
            self.result["Accuracy"] = acc
            self.result["Cross Validated Accuracy"] = cv_acc
            self.result["Model"] = m_list
        elif self.task == 'anomaly':
            # self.result_anom[model_name] = [self.accuracy, self.m_absolute_error, self.rm_squared_error, self.model]
            self.result_anom = []
        return self.result if self.task != 'anomaly' else self.result_anom

    def _stacking_ensembler(self):
        if self.task == 'regression':
            stacking_model = StackingRegressor(
                regressors=self.models,
                meta_regressor=LinearRegression(),
                verbose=self.verbose
            )
        elif self.task == 'classification':
            stacking_model = StackingClassifier(
                classifiers=self.models,
                meta_classifier=LogisticRegression(),
                verbose=self.verbose,

            )
        stacking_model.fit(self.X_train, self.y_train)
        return stacking_model

    def _stacking_ensembler_cv(self):
        if self.task == 'regression':
            stacking_model = StackingCVRegressor(
                regressors=self.models,
                meta_regressor=LinearRegression(),
                verbose=self.verbose,
                cv=self.cv_folds,
                random_state=self.random_state,
                refit=False,
            )
        elif self.task == 'classification':
            stacking_model = StackingCVClassifier(
                classifiers=self.models,
                meta_classifier=LogisticRegression(),
                verbose=self.verbose,
                random_state=self.random_state,
                n_jobs=-1,
            )
        stacking_model.fit(self.X_train, self.y_train)
        return stacking_model

    def _max_voting(self):
        if self.task == 'regression':
            ensembler = VotingRegressor(
                estimators=self.est, n_jobs=-1, verbose=self.verbose)
            ensembler.fit(self.X_train, self.y_train)
        elif self.task == 'classification':
            ensembler = EnsembleVoteClassifier(
                clfs=self.models, verbose=self.verbose)
            ensembler.fit(self.X_train, self.y_train)
        return ensembler

    def evaluator(self, model, model_name):
        self.y_pred = model.predict(self.X_val)
        if self.task == 'regression':
            self.kfold_accuracy = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv_folds, scoring='r2')
            self.accuracy = r2_score(self.y_val, self.y_pred)
            self.mae = mean_absolute_error(
                self.y_val, self.y_pred)
            self.rmse = mean_squared_error(
                self.y_val, self.y_pred, squared=False
            )
        elif self.task == 'classification':
            self.kfold_accuracy = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv_folds, scoring='accuracy')
            self.accuracy = accuracy_score(self.y_val, self.y_pred)
