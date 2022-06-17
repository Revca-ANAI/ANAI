import traceback

from anai.utils.explainable_anai.permutation import permutational_feature_importance
from anai.utils.explainable_anai.shap import shap_feature_importance
from anai.utils.explainable_anai.surrogate import surrogate_decision_tree
from colorama import Fore


class Explainer:
    def __init__(self):
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.dataset = None
        self.cv_folds = None
        self.fit_params = None
        self.isReg = None
        self.columns = None
        self.y_test = None

    def set_params(
        self,
        features,
        X_train,
        X_test,
        y_train,
        y_test,
        cv_folds=10,
        fit_params={},
        isReg=True,
        columns=None,
    ):
        self.features = features
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.fit_params = fit_params
        self.isReg = isReg
        self.columns = columns

    def permutation(self, model):
        try:
            permutational_feature_importance(
                self.features.columns, self.X_train, self.y_train, model, self.isReg
            )
        except Exception as e:
            print(Fore.YELLOW + "Automatically switching to Surrogate mode\n")
            try:
                permutational_feature_importance(
                    self.features.columns,
                    self.X_train,
                    self.y_train,
                    surrogate_decision_tree(
                        model, self.X_train, self.y_train, isReg=self.isReg
                    ),
                    self.isReg,
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(Fore.RED + "Explaining ANAI using Permutations Failed [X]\n")
        print(Fore.GREEN + "Explaining ANAI Done [", "\u2713", "]\n")

    def shap(self, model):
        try:
            shap_feature_importance(self.features.columns, self.X_train, model)
        except Exception as e:
            print(Fore.YELLOW + "Automatically switching to Surrogate mode\n")
            try:
                shap_feature_importance(
                    self.features.columns,
                    self.X_train,
                    surrogate_decision_tree(model, self.X_train, isReg=self.isReg),
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print(Fore.RED + "Explaining ANAI using SHAP Failed [X]\n")
        print(Fore.GREEN + "Explaining ANAI Done [", "\u2713", "]\n")
