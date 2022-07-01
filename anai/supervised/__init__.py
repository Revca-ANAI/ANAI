import copy
import os
import time
import traceback
import warnings
from pickle import dump, load

import modin.pandas as pd
import pandas
import numpy as np
import optuna
from anai.preprocessing import Preprocessor
from anai.supervised import *
from anai.utils.best import Best
from anai.utils.configs import *
from anai.utils.connectors import load_data_from_config
from anai.utils.connectors.data_handler import df_loader
from anai.utils.ensembler import Ensembler
from anai.utils.explainable_anai.explain_core import Explainer
from anai.utils.handlers.dimension_handler import DimensionHandler
from anai.utils.plot import confusion_matrix_plot
from anai.utils.predictors import (classification_predictor,
                                   regression_predictor)
from anai.utils.tuner.anai_tuner import anai_tuner
from anai.utils.validator import *
from colorama import Fore
from IPython.display import display
from joblib import dump, load
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)

warnings.filterwarnings("ignore")


class Classification:
    def __init__(
        self,
        df=None,
        target: str = None,
        filepath: str = None,
        config: bool = False,
        except_columns: list = [],
        predictor: list =["lr"],
        params: dict ={},
        tune: bool =False,
        test_size: float =0.2,
        cv_folds: int =10,
        random_state: int =42,
        pca_kernel: str ="linear",
        n_components_lda: int=1,
        lda: bool =False,
        pca: bool =False,
        n_components_pca: int =2,
        smote: bool = "False",
        k_neighbors: int=1,
        verbose:bool=False,
        exclude_models:list=[],
        path:str=None,
        optuna_sampler=TPESampler(multivariate=True),
        optuna_direction:str="maximize",
        optuna_n_trials:int=100,
        metric:str="accuracy",
        lgbm_objective:str="binary",
        ensemble:bool=True,
    ):
        """Initializes the Classifier class.

        Parameters:

            features : array
                        features array
            lables : array
                        labels array
            except_columns (list): [List of Columns to be excluded from the dataset]
            predictor : list
                        Predicting model to be used
                        Default ['lr']  - Logistic Regression\n
                        Available Predictors:
                            "lr": "Logistic Regression",\n
                            "sgd": "Stochastic Gradient Descent",\n
                            "perc": "Perceptron",\n
                            "pass": "Passive Aggressive Classifier",\n
                            "ridg": "Ridge Classifier",\n
                            "svm": "Support Vector Machine",\n
                            "knn": "K-Nearest Neighbors",\n
                            "dt": "Decision Trees",\n
                            "nb": "Naive Bayes",\n
                            "rfc": "Random Forest Classifier",\n
                            "gbc": "Gradient Boosting Classifier",\n
                            "ada": "AdaBoost Classifier",\n
                            "bag": "Bagging Classifier",\n
                            "ext": "Extra Trees Classifier",\n
                            "lgbm": "LightGBM Classifier",\n
                            "cat": "CatBoost Classifier",\n
                            "xgb": "XGBoost Classifier",\n
                            "ann": "Multi Layer Perceptron Classifier",\n
                            "poisson": "Poisson Classifier",\n
                            "huber": "Huber Classifiers",\n
                            "ridge_cv": "RidgeCV Classifier",\n
                            "encv": "ElasticNet CV Classifier",\n
                            "lcv": "LassoCV Classifier",\n
                            "llic": "LassoLarsIC Classifier",\n
                            "llcv": "LassoLarsCV Classifier",\n
                            "all": "All Classifiers",\n
            params : dict
                        contains parameters for model
            tune : boolean
                    when True Applies GridSearch CrossValidation
                    Default is False
            test_size: float or int, default=.2
                        If float, should be between 0.0 and 1.0 and represent
                        the proportion of the dataset to include in
                        the test split.
                        If int, represents the absolute number of test samples.
            cv_folds : int
                    No. of cross validation folds. Default = 10
            pca : str
                if 'y' will apply PCA on Train and Validation set. Default = 'n'
            lda : str
                if 'y' will apply LDA on Train and Validation set. Default = 'n'
            pca_kernel : str
                    Kernel to be use in PCA. Default = 'linear'
            n_components_lda : int
                    No. of components for LDA. Default = 1
            n_components_pca : int
                    No. of components for PCA. Default = 2
            loss : str
                    loss method for ann. Default = 'binary_crossentropy'
                    rate for dropout layer. Default = 0
            smote : str,
                Whether to apply SMOTE. Default = 'y'
            k_neighbors : int
                No. of neighbors for SMOTE. Default = 1
            verbose : boolean
                Verbosity of models. Default = False
            exclude_models : list
                List of models to be excluded when using predictor = 'all' . Default = []
            path : list
                List containing path to saved model and scaler. Default = None
                Example: [model.pkl, scaler.pkl]
            random_state : int
                Random random_state for reproducibility. Default = 42
            optuna_sampler : Function
                Sampler to be used in optuna. Default = TPESampler()
            optuna_direction : str
                Direction of optimization. Default = 'maximize'
                Available Directions:
                    maximize : Maximize
                    minimize : Minimize
            optuna_n_trials : int
                No. of trials for optuna. Default = 100
            optuna_metric: str
                Metric to be used in optuna. Default = 'R^2'
            lgbm_objective : str
                Objective for lgbm classifier. Default = 'binary'
            ensemble : boolean
                Whether to use ensemble. Default = True

        Returns:

            Dict Containing Name of Classifiers, Its K-Fold Cross Validated Accuracy and Prediction set

            Dataframe containing all the models and their accuracies when predictor is 'all'

        Example:

            import anai
            ai = anai.run(
                        filepath='examples/test_data.csv', 
                        target='PE',
                        predictor=['lr'],
            )

        """
        print(Fore.MAGENTA + intro, "\n")
        print(Fore.GREEN + "Started ANAI [", "\u2713", "]\n")
        if config:
            print(Fore.YELLOW + "Searching for ANAI Config File [*]", "\n")
            time.sleep(1)
            if os.path.exists(os.getcwd() + "/anai_config.yaml"):
                print(Fore.GREEN + "ANAI Config File Found [", "\u2713", "]\n")
                df, target = load_data_from_config(
                    os.getcwd() + "/anai_config.yaml")
            else:
                raise FileNotFoundError("ANAI Config File Not Found")
        if df is None:
            if filepath is not None:
                df = df_loader(filepath)
            else:
                raise ValueError("Please provide a dataframe or a filepath")
        if type(predictor) == list:
            if not "all" in predictor:
                self.predictor = predictor[0] if len(
                    predictor) == 1 else predictor
            else:
                self.predictor = predictor
        else:
            self.predictor = predictor
        bool_pred, pred = pred_check(predictor, pred_type="classification")
        if not bool_pred:
            raise ValueError(unsupported_pred_warning.format(pred))
        self.df = df
        self.data_filepath = filepath
        if target is None:
            raise ValueError("Please provide a target variable")
        self.except_columns = except_columns
        self.target = target
        self.preprocessor = Preprocessor(dataset=df, target=target)
        self.features = None
        self.labels = None
        self.ori_features = None
        self.original_predictor = predictor
        self.params = params
        self.tune = tune
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pca_kernel = pca_kernel
        self.n_components_lda = n_components_lda
        self.lda = lda
        self.pca = pca
        self.n_components_pca = n_components_pca

        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.exclude_models = exclude_models
        self.sampler = optuna_sampler
        self.direction = optuna_direction
        self.n_trials = optuna_n_trials
        self.metric = metric
        self.lgbm_objective = lgbm_objective

        self.accuracy_scores = {}
        self.reg_result = {}
        self.accuracy = 0
        self.y_pred = []
        self.kfold_accuracy = 0
        self.classifier_name = ""
        self.sc = 0

        self.k_fold_accuracy = []
        self.acc = []
        self.bestacc = []
        self.best_params = []
        self.tuned_trained_model = []
        self.best_classifier_path = ""
        self.scaler_path = ""
        self.classifier_model = []
        self.result_df = pd.DataFrame(index=None)
        self.classifiers = copy.deepcopy(classifiers)
        self.model = None
        
        self.ensemble = ensemble
        
        for i in self.exclude_models:
            self.classifiers.pop(i)
        self.classifier_wrap = None
        self.best_classifier = "First Run the Predictor in All mode"
        self.objective = None
        self.pred_mode = ""
        self.model_to_predict = []

        if path != None:
            try:
                self.model, self.sc = self.__load(path)
            except Exception as e:
                print(Fore.RED + e)
                print(Fore.RED + "Model not found")
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.explainer = Explainer()
        self.ensembler = Ensembler(
            "classification",
            n_estimators=3,
            n_clusters=3,
            estimators=None,
            verbose=False,
            random_state=self.random_state
        )
        self.fit_params = {}
        self.encoded_column_names = []
        self.dimension_handler = DimensionHandler()
        self.encoder = None
        self.le_encoder = None
        self.__fit()

    def __fit(self):
        """[Takes Features and Labels and Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
        scales X_train, self.X_val using StandardScaler.
        Fits every model on training set and predicts results,
        finds accuracy of model applies K-Fold Cross Validation
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
        Applies GridSearch Cross Validation and gives best params out from param list.]

        Args:
            features ([Pandas DataFrame]): [DataFrame containing Features]
            labels ([Pandas DataFrame]): [DataFrame containing Labels]
        """
        # Time Function ---------------------------------------------------------------------

        self.start = time.time()

        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            try:
                type_check(self.df)
            except TypeError as e:
                print(Fore.RED + "[{}]: {}".format(type(e).__name__, e))
                self.end = time.time()
                final_time = self.end - self.start
                print(Fore.RED + "\nANAI Run Failed [", "\u2713", "]\n")
                print(Fore.BLUE + "Time Elapsed : ",
                      f"{final_time:.2f}", "seconds \n")
                return

            print(Fore.YELLOW + "Preprocessing Started [*]\n")
            if self.df.isna().any().any():
                print(Fore.YELLOW + "Imputing Missing Values by mean [*]\n")
                self.df = self.preprocessor.impute('mean')
                print(Fore.GREEN + "Imputing Done [", "\u2713", "]\n")
            self.features = self.df.drop(
                [self.target] + self.except_columns, axis=1)
            self.labels = self.df[self.target]
            self.ori_features = self.features.copy()
            (
                self.features,
                self.labels,
                self.encoded_column_names,
                self.encoder,
                self.le_encoder,
            ) = self.preprocessor.encode(
                type="anai", features=self.features, labels=self.labels
            )

            self.features, self.labels = sparse_check(
                self.features, self.labels)
            (
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.sc,
            ) = self.preprocessor.prepare(
                self.features,
                self.labels,
                self.test_size,
                self.random_state,
                self.smote,
                self.k_neighbors,
            )

            self.X_train, self.X_val = self.dimension_handler.dimensionality_reduction(
                self.lda,
                self.pca,
                self.X_train,
                self.X_val,
                self.y_train,
                self.n_components_lda,
                self.n_components_pca,
                self.pca_kernel,
                self.start,
            )

        print(Fore.GREEN + "Preprocessing Done [", "\u2713", "]\n")
        if self.original_predictor == "all" or type(self.predictor) == list:
            if "all" in self.predictor and type(self.predictor) == list:
                self.predictor.remove("all")
            self.model_to_predict = (
                self.predictor
                if len(self.predictor) > 1 and type(self.predictor) == list
                else self.classifiers
            )

            self.result_df["Name"] = (
                list(self.classifiers[i] for i in self.predictor)
                if type(self.predictor) == list and len(self.predictor) > 1
                else list(self.classifiers.values())
            )
            self.pred_mode = (
                "all"
                if len(self.predictor) > 1 and type(self.predictor) == list
                else "single"
            )
            self.__fitall()
            return
        self.model, self.objective = classification_predictor(
            self.predictor,
            self.params,
            self.X_train,
            self.y_train,
            self.cv_folds,
            self.random_state,
            self.metric,
            verbose=self.verbose,
            lgbm_objective=self.lgbm_objective,
        )
        try:
            self.model.fit(
                self.X_train, self.y_train
            ) if not self.predictor == "tabnet" else self.model.fit(
                self.X_train,
                self.y_train,
                patience=3,
                max_epochs=self.epochs,
                batch_size=self.batch_size,
            )
            self.fit_params = {}
        except Exception as error:
            print(traceback.format_exc())
            print(Fore.RED + "Classifier Build Failed with error: ", error, "\n")
            return
        print(Fore.GREEN + "Model Trained Successfully [", "\u2713", "]\n")
        try:
            print(Fore.YELLOW + "Evaluating Model Performance [*]\n")
            self.y_pred = self.model.predict(self.X_val)
            if (
                self.predictor == "ann"
            ):
                self.y_pred = (self.y_pred > 0.5).astype("int32")
            self.accuracy = accuracy_score(self.y_val, self.y_pred)
            print(
                Fore.CYAN
                + "        Validation Accuracy is : {:.2f} %".format(
                    self.accuracy * 100
                )
            )
            self.classifier_name, self.kfold_accuracy = cv(
                self.model,
                self.predictor,
                self.X_train,
                self.y_train,
                self.cv_folds,
                verbose=self.verbose,
            )
            if self.features.shape[1] < 8:
                confusion_matrix_plot(self.y_pred, self.y_val)
        except Exception as error:
            print(Fore.RED + "Model Evaluation Failed with error: ", error, "\n")
        finally:
            print(Fore.GREEN + "Model Evaluation Completed [", "\u2713", "]\n")

        if not self.predictor == "nb" and self.tune:
            self.__tuner()
        print(Fore.GREEN + "Completed ANAI Run [", "\u2713", "]\n")
        self.end = time.time()
        final_time = self.end - self.start
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")

    def __fitall(self):
        print(Fore.YELLOW + "Training ANAI [*]\n")
        if self.params != {}:
            warnings.warn(params_use_warning, UserWarning)
            self.params = {}
        for _, self.predictor in enumerate(self.model_to_predict):
            if not self.predictor in self.exclude_models:
                try:
                    (
                        self.model,
                        self.objective,
                    ) = classification_predictor(
                        self.predictor,
                        self.params,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        self.random_state,
                        self.metric,
                        mode="multi",
                        verbose=self.verbose,
                        lgbm_objective=self.lgbm_objective,
                    )
                except Exception as error:
                    print(traceback.format_exc())
                    print(
                        Fore.RED + classifiers[self.predictor],
                        "Model Train Failed with error: ",
                        error,
                        "\n",
                    )
                try:
                    self.model.fit(
                        self.X_train, self.y_train
                    )
                    self.fit_params = {}
                    self.y_pred = self.model.predict(self.X_val)
                    if (
                        self.predictor == "ann"
                    ):
                        self.y_pred = (self.y_pred > 0.5).astype("int32")
                    self.accuracy = accuracy_score(self.y_val, self.y_pred)
                    self.acc.append(self.accuracy * 100)
                    self.classifier_name, self.kfold_accuracy = cv(
                        self.model,
                        self.predictor,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        all_mode=True,
                    )
                    self.k_fold_accuracy.append(self.kfold_accuracy)
                    self.classifier_model.append(self.model)
                except Exception as error:
                    print(traceback.format_exc())
                    print(
                        classifiers[self.predictor],
                        "Evaluation Failed with error: ",
                        error,
                        "\n",
                    )
                if self.tune:
                    self.__tuner(all_mode=True, single_mode=False)

        self.result_df["Accuracy"] = self.acc
        self.result_df["Cross Validated Accuracy"] = self.k_fold_accuracy
        self.result_df["Model"] = self.classifier_model
        self.result_df = self.result_df.sort_values(
            by="Cross Validated Accuracy", ascending=False)
        top_result = self.result_df.sort_values(
            by=['Cross Validated Accuracy'], ascending=False).head(5)
        if self.ensemble:
            estimators = []
            est1 = []
            for i in range(top_result.shape[0]):
                if not top_result.iloc[i]['Name'] == 'K-Nearest Neighbors':
                    estimators.append(
                        top_result.iloc[i]['Model'])
                    est1.append(
                        (top_result.iloc[i]['Name'], top_result.iloc[i]['Model']))
            print(Fore.YELLOW + "Ensembling on top {} models\n".format(
                5 if len(estimators) > 5 else len(estimators)))
            try:
                ens_result = self.ensembler.ensemble(
                    self.X_train, self.y_train, self.X_val, self.y_val, cv_folds=self.cv_folds, estimators=estimators, est=est1)
                self.result_df = pd.concat(
                    [self.result_df, ens_result], axis=0)
            except Exception as error:
                print(traceback.format_exc())
                print(Fore.RED+"Ensembling Failed with error: ", error, "\n")
        self.result_df = self.result_df.sort_values(
            by=['Cross Validated Accuracy'], ascending=False).reset_index(drop=True)
        print(Fore.GREEN + "Training Done [", "\u2713", "]\n")
        print(Fore.CYAN + "Results Below\n")
        if self.tune:
            self.result_df["Best Parameters"] = self.best_params
            self.result_df["Best Accuracy"] = self.bestacc
            self.best_classifier = Best(
                self.result_df.loc[self.result_df["Best Accuracy"].idxmax()],
                self.tune,
            )
        else:
            self.best_classifier = Best(
                self.result_df.loc[self.result_df["Cross Validated Accuracy"].idxmax(
                )], self.tune
            )
        display(self.result_df.drop(['Model'], axis=1))
        print(Fore.GREEN + "\nCompleted ANAI Run [", "\u2713", "]\n")
        if len(self.model_to_predict) > 1:
            self.model = self.best_classifier.model
            self.model_name = self.best_classifier.name
            self.end = time.time()
            final_time = self.end - self.start
            self.meta_path = self.save(
                best=True
            )
            print(
                Fore.CYAN
                + "Saved Best Model at {} ".format(
                    self.meta_path
                ),
                "\n",
            )
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")
        return

    def __tuner(self, all_mode=False, single_mode=False):
        if not all_mode:
            print(Fore.YELLOW + "Tuning Started [*]\n")
        if not self.predictor == "nb":
            (
                best_params,
                self.best_accuracy,
                self.best_trained_model,
            ) = anai_tuner(
                self.predictor,
                self.objective,
                self.n_trials,
                self.sampler,
                self.direction,
                self.X_train,
                self.y_train,
                self.cv_folds,
                self.random_state,
                self.metric,
                all_mode=all_mode,
            )
        if self.predictor == "nb":
            self.best_params = "Not Applicable"
            self.best_accuracy = 0
        self.best_params.append(self.best_params)
        self.bestacc.append(self.best_accuracy * 100)
        self.tuned_trained_model.append(self.best_trained_model)
        if not all_mode or single_mode:
            print(Fore.GREEN + "Tuning Done [", "\u2713", "]\n")

    def result(self):
        """[Makes a dictionary containing Classifier Name, K-Fold CV Accuracy, RMSE, Prediction set.]

        Returns:
            [dict]: [Dictionary containing :
                        - "Classifier" - Classifier Name
                        - "Accuracy" - Cross Validated CV Accuracy
                        - "YPred" - Array for Prediction set
                        ]
            [dataframe] : [Dataset containing accuracy and best_params
                            for all predictors only when predictor = 'all' is used
                            ]
        """
        if not self.pred_mode == "all":
            self.reg_result["Classifier"] = self.classifier_name
            self.reg_result["Accuracy"] = self.kfold_accuracy
            self.reg_result["YPred"] = self.y_pred
            reg_result = pd.DataFrame.from_dict(
                self.reg_result, orient='index', columns=['Summary'])
            return reg_result
        if self.pred_mode == "all":
            return self.result_df

    def predict(self, X_test):
        """[Takes test set and returns predictions for that test set]

        Args:
            X_test ([Array]): [Array Containing Test Set]

        Returns:
            [Array]: [Predicted set for given test set]
        """
        if self.pred_mode == "all":
            classifier = copy.deepcopy(self.best_classifier.model)
            print(Fore.YELLOW + "Predicting on Test Set using Best Model[*]\n")
        else:
            classifier = copy.deepcopy(
                self.model
            )
        X_test = np.array(X_test) if type(X_test) == list else X_test
        if isinstance(X_test, pandas.core.frame.DataFrame) or isinstance(X_test, modin.pandas.DataFrame):
            if self.target in X_test.columns:
                X_test = X_test.drop(self.target, axis=1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test = pd.DataFrame(X_test, columns=self.features.columns) if type(
            X_test) == np.ndarray else X_test
        if isinstance(X_test, modin.pandas.DataFrame):
            X_test = X_test._to_pandas()
        y_test = classifier.predict(
            self.sc.transform(self.encoder.transform(X_test)))
        print(Fore.BLUE + "Predicted Value : ", y_test, "\n")
        print(Fore.GREEN + "Prediction Done [", "\u2713", "]\n")
        return y_test

    def save(self, path=None, best=False, **kwargs):
        """
        Saves the model and its scaler to a file provided with a path.
        If no path is provided will create a directory named
        anai_info/models/ and anai_info/scaler/ in current working directory
        Args:
            path ([list]): [List containing path to save the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            Path to the saved model and its scaler.
        """
        if not type(path) == list and path != None:
            raise TypeError("Path must be a list")
        if self.pred_mode == "all" and best == False:
            raise TypeError("Cannot save model for all predictors")
        dir_path_model = path[0] if path else "anai_info/models/classifier/"
        dir_path_scaler = path[1] if path else "anai_info/scalers/classifier/"
        model_name = classifiers[self.predictor].replace(" ", "_")
        if best:
            dir_path_model = "anai_info/best/classifier/models/"
            dir_path_scaler = "anai_info/best/classifier/scalers/"
            model_name = self.best_classifier.name.replace(" ", "_")
        os.makedirs(dir_path_model, exist_ok=True)
        os.makedirs(dir_path_scaler, exist_ok=True)
        timestamp = str(int(time.time()))
        path_model = dir_path_model + model_name + "_" + timestamp + ".pkl"
        path_scaler = (
            dir_path_scaler + model_name + "_" + "Scaler" + "_" + timestamp + ".pkl"
        )
        if (
            not kwargs.get("model")
            and not kwargs.get("best")
            and not kwargs.get("scaler")
        ):
            dump(self.model, open(path_model, "wb"))
            dump(self.sc, open(path_scaler, "wb"))
        else:
            dump(kwargs.get("model"), open(path_model, "wb"))
            dump(kwargs.get("scaler"), open(path_scaler, "wb"))
        if not best:
            print("Model Saved at {} and Scaler at {}".format(
                path_model, path_scaler))
        return path_model, path_scaler

    def __load(self, path=None):
        """
        Loads model and scaler from the specified path
        Args:
            path ([list]): [List containing path to load the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            [Model] : [Loaded model]
            [Scaler] : [Loaded scaler]
        """

        model_path = path[0] if path[0] else None
        scaler_path = path[1] if path[1] else None
        if not ".pkl" in model_path and not model_path == None:
            raise TypeError(
                "[Error] Model Filetype not supported. Please use .pkl type "
            )
        if not ".pkl" in scaler_path and not scaler_path == None:
            raise TypeError(
                "[Error] Scaler Filetype not supported. Please use .pkl type "
            )
        if model_path != None and scaler_path != None:
            model = load(open(model_path, "rb"))
            scaler = load(open(scaler_path, "rb"))
            print(
                Fore.GREEN
                + "[Info] Model and Scaler Loaded from {} and {}".format(
                    model_path, scaler_path
                )
            )
            return model, scaler
        elif model_path != None and scaler_path == None:
            model = load(open(model_path, "rb"))
            print(Fore.GREEN +
                  "[Info] Model Loaded from {}".format(model_path))
            return model
        elif model_path == None and scaler_path != None:
            scaler = load(open(scaler_path, "rb"))
            print(Fore.GREEN +
                  "[Info] Scaler Loaded from {}".format(scaler_path))
            return scaler
        else:
            raise ValueError("No path specified.Please provide actual path\n")

    def explain(self, method):
        """
        Returns the importance features of the dataset
        """
        columns = self.features.columns
        self.explainer.set_params(
            self.features,
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            self.cv_folds,
            self.fit_params,
            False,
            columns,
        )
        if self.pred_mode == "all":
            classifier = copy.deepcopy(self.best_classifier.model)
            print(Fore.YELLOW + "Explaining Best ANAI model [*]\n")
        else:
            classifier = copy.deepcopy(
                self.model
            )
            print(Fore.YELLOW + "Explaining ANAI [*]\n")
        if self.original_predictor == "all":
            raise TypeError(
                "[Error] This method is only applicable on single predictor"
            )
        elif method == "perm":
            self.explainer.permutation(model=classifier)
        elif method == "shap":
            self.explainer.shap(model=classifier)
        else:
            raise NotImplementedError(
                "Technique not implemented. Please choose from perm, shap"
            )


class Regression:
    def __init__(
        self,
        df=None,
        target: str = None,
        filepath: str = None,
        config: bool = False,
        except_columns: list = [],
        predictor: list = ["lin"],
        params: dict = {},
        tune: bool = False,
        test_size: float = 0.2,
        cv_folds: int = 10,
        random_state: int = 42,
        pca_kernel: str = "linear",
        n_components_lda: int = 1,
        lda: bool = False,
        pca: bool = False,
        n_components_pca: int = 2,
        smote: bool = "False",
        k_neighbors: int = 1,
        verbose: bool = False,
        exclude_models: list = [],
        path: str = None,
        optuna_sampler=TPESampler(multivariate=True),
        optuna_direction: str = "maximize",
        optuna_n_trials: int = 100,
        metric: str = "r2",
        ensemble: bool = True,
    ):
        """Initializes the Regression class

        Parameters:
            df (dataframe): [Dataset containing features and target]
            target (str): [Target Column Name]
            except_columns (list): [List of Columns to be excluded from the dataset]
            predictor : list
                        Predicting models to be used
                        Default ['lin'] - 'Linear Regression'\n
                        Available Predictors:
                            "lin": "Linear Regression",\n
                            "sgd": "Stochastic Gradient Descent Regressor",\n
                            "krr": "Kernel Ridge Regression",\n
                            "elas": "Elastic Net Regression",\n
                            "br": "Bayesian Ridge Regression",\n
                            "svr": "Support Vector Regressor",\n
                            "knn": "K-Nearest Neighbors",\n
                            "dt": "Decision Trees Regressor",\n
                            "rfr": "Random Forest Regressor",\n
                            "gbr": "Gradient Boosted Regressor",\n
                            "ada": "AdaBoostRegressor",\n
                            "bag": "Bagging Regressor",\n
                            "ext": "Extra Trees Regressor",\n
                            "lgbm": "LightGBM Regressor",\n
                            "xgb": "XGBoost Regressor",\n
                            "cat": "Catboost Regressor",\n
                            "ann": "Multi-Layer Perceptron Regressor",\n
                            "poisson": "Poisson Regressor",\n
                            "huber": "Huber Regressor",\n
                            "gamma": "Gamma Regressor",\n
                            "ridge": "Ridge CV Regressor",\n
                            "encv": "ElasticNetCV Regressor",\n
                            "lcv": "LassoCV Regressor",\n
                            "llic": "LassoLarsIC Regressor",\n
                            "llcv": "LassoLarsCV Regressor",\n
                            "all": "All Regressors",\n
            params : dict
                        contains parameters for model
            tune : boolean
                    when True Applies Optuna to find best parameters for model
                    Default is False
            test_size: float or int, default=.2
                        If float, should be between 0.0 and 1.0 and represent
                        the proportion of the dataset to include in
                        the test split.
                        If int, represents the absolute number of test samples.
            cv_folds : int
                    No. of cross validation folds. Default = 10
            pca : bool
                if True will apply PCA on Train and Validation set. Default = False
            lda : str
                if True will apply LDA on Train and Validation set. Default = False
            pca_kernel : str
                    Kernel to be use in PCA. Default = 'linear'
            n_components_lda : int
                    No. of components for LDA. Default = 1
            n_components_pca : int
                    No. of components for PCA. Default = 2
            smote : Bool,
                Whether to apply SMOTE. Default = True
            k_neighbors : int
                No. of neighbors for SMOTE. Default = 1
            verbose : boolean
                Verbosity of models. Default = False
            exclude_models : list
                List of models to be excluded when using predictor = 'all' . Default = []
            path : list
                List containing path to saved model and scaler. Default = None
                Example: [model.pkl, scaler.pkl]
            random_state : int
                Random random_state for reproducibility. Default = 42
            optuna_sampler : Function
                Sampler to be used in optuna. Default = TPESampler()
            optuna_direction : str
                Direction of optimization. Default = 'maximize'
                Available Directions:
                    maximize : Maximize
                    minimize : Minimize
            optuna_n_trials : int
                No. of trials for optuna. Default = 100
            ensemble : boolean
                Whether to use ensemble methods. Default = True

        Returns:

            Dict Containing Name of Regressor, Its K-Fold Cross Validated Accuracy, RMSE, Prediction set

            Dataframe containing all the models and their accuracies when predictor is 'all'

        Example:
            import anai
            
            ai = anai.run(
                        filepath='examples/Folds5x2_pp.xlsx', 
                        target='PE',
                        predictor=['lin'],
            )

        """
        print(Fore.MAGENTA + intro, "\n")
        print(Fore.GREEN + "Started ANAI [", "\u2713", "]\n")
        if config:
            print(Fore.YELLOW + "Searching for ANAI Config File [*]", "\n")
            time.sleep(1)
            if os.path.exists(os.getcwd() + "/anai_config.yaml"):
                print(Fore.GREEN + "ANAI Config File Found [", "\u2713", "]\n")
                df, target = load_data_from_config(
                    os.getcwd() + "/anai_config.yaml")
            else:
                raise FileNotFoundError("ANAI Config File Not Found")
        if df is None:
            if filepath is not None:
                df = df_loader(filepath)
            # elif config_filepath is not None:
            #     df, target = load_data_from_config(config_filepath)
            else:
                raise ValueError("Please provide a dataframe or a filepath")
        if type(predictor) == list:
            if not "all" in predictor:
                self.predictor = predictor[0] if len(
                    predictor) == 1 else predictor
            else:
                self.predictor = predictor
        else:
            self.predictor = predictor
        self.df = df
        self.data_filepath = filepath
        if target is None:
            raise ValueError("Please provide a target variable")
        self.target = target
        bool_pred, pred = pred_check(predictor, pred_type="regression")
        if not bool_pred:
            raise ValueError(unsupported_pred_warning.format(pred))
        self.except_columns = except_columns
        self.preprocessor = Preprocessor(dataset=df, target=target)
        self.original_predictor = predictor
        self.params = params
        self.tune = tune
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pca_kernel = pca_kernel
        self.n_components_lda = n_components_lda
        self.lda = lda
        self.pca = pca
        self.n_components_pca = n_components_pca
        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.exclude_models = exclude_models
        self.sampler = optuna_sampler
        self.direction = optuna_direction
        self.n_trials = optuna_n_trials
        self.metric = metric

        self.accuracy_scores = {}
        self.reg_result = {}
        self.rm_squared_error = 0
        self.accuracy = 0
        self.y_pred = []
        self.kfold_accuracy = 0
        self.regressor_name = ""
        self.sc = 0

        self.k_fold_accuracy = []
        self.acc = []
        self.mae = []
        self.rmse = []
        self.bestacc = []
        self.best_params = []
        self.regressor_model = []
        self.tuned_trained_model = []
        self.best_regressor_path = ""
        self.scaler_path = ""

        self.result_df = pd.DataFrame(index=None)
        self.regressors = copy.deepcopy(regressors)
        for i in self.exclude_models:
            self.regressors.pop(i)
        self.regressor_wrap = None
        self.best_regressor = "First Run the Predictor in All mode"
        self.objective = None
        self.pred_mode = ""
        self.model_to_predict = None
        self.model = None
        
        self.ensemble = ensemble
        
        if path != None:
            try:
                self.model, self.sc = self.__load(path)
            except Exception as e:
                print(Fore.RED + e)
                print(Fore.RED + "Model not found")
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.explainer = Explainer()
        self.ensembler = Ensembler(
            "regression",
            n_estimators=3,
            n_clusters=3,
            estimators=None,
            verbose=False,
            random_state=self.random_state
        )
        self.fit_params = {}
        self.dimension_handler = DimensionHandler()
        self.encoder = None
        self.features = None
        self.labels = None
        self.__fit()

    def __fit(self):
        """[Takes Features and Labels and Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
        scales X_train, X_val using StandardScaler.
        Fits model on training set and predicts results, Finds R^2 Scoreand mean square error
        finds accuracy of model applies K-Fold Cross Validation
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
        Applies GridSearch Cross Validation and gives best params out from param list.]
        """

        # Time Function ---------------------------------------------------------------------

        self.start = time.time()
        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            try:
                type_check(self.df)
            except TypeError as e:
                print(Fore.RED + "[{}]: {}".format(type(e).__name__, e))
                self.end = time.time()
                final_time = self.end - self.start
                print(Fore.RED + "\nANAI Run Failed [", "\u2713", "]\n")
                print(Fore.BLUE + "Time Elapsed : ",
                      f"{final_time:.2f}", "seconds \n")
                return

            print(Fore.YELLOW + "Preprocessing Started [*]\n")
            if self.df.isna().any().any():
                print(Fore.YELLOW + "Imputing Missing Values by mean [*]\n")
                self.df = self.preprocessor.impute('mean')
                print(Fore.GREEN + "Imputing Done [", "\u2713", "]\n")
            self.features = self.df.drop([self.target] + self.except_columns, axis=1)
            self.labels = self.df[self.target]
            self.features, self.labels, _, self.encoder, _ = self.preprocessor.encode(
                type="anai", features=self.features, labels=self.labels
            )
            self.features, self.labels = sparse_check(
                self.features, self.labels)
            (
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.sc,
            ) = self.preprocessor.prepare(
                self.features,
                self.labels,
                self.test_size,
                self.random_state,
                self.smote,
                self.k_neighbors,
            )
            self.X_train, self.X_val = self.dimension_handler.dimensionality_reduction(
                self.lda,
                self.pca,
                self.X_train,
                self.X_val,
                self.y_train,
                self.n_components_lda,
                self.n_components_pca,
                self.pca_kernel,
                self.start,
            )

        print(Fore.GREEN + "Preprocessing Done [", "\u2713", "]\n")

        if self.original_predictor == "all" or type(self.predictor) == list:
            if "all" in self.predictor and type(self.predictor) == list:
                self.predictor.remove("all")
            self.model_to_predict = (
                self.predictor
                if len(self.predictor) > 1 and type(self.predictor) == list
                else self.regressors
            )
            self.result_df["Name"] = (
                list(self.regressors[i] for i in self.predictor)
                if type(self.predictor) == list and len(self.predictor) > 1
                else list(self.regressors.values())
            )
            self.pred_mode = (
                "all"
                if type(self.predictor) == list and len(self.predictor) > 1
                else "single"
            )
            self.__fitall()
            return

        self.model, self.objective = regression_predictor(
            self.predictor,
            self.params,
            self.X_train,
            self.y_train,
            self.cv_folds,
            self.random_state,
            self.metric,
            mode=self.pred_mode,
            verbose=self.verbose,
        )
        try:
            if self.predictor:
                if hasattr(self.model.fit, "verbose"):
                    self.model.fit(
                        self.X_train, self.y_train, verbose=0 if not self.verbose else 1
                    )
                else:
                    self.model.fit(self.X_train, self.y_train)
                self.fit_params = {}
        except Exception as error:
            print(traceback.format_exc())
            print(Fore.RED + "Regressor Build Failed with error: ", error, "\n")
        finally:
            print(Fore.GREEN + "Model Trained Successfully [", "\u2713", "]\n")

        try:
            print(Fore.YELLOW + "Evaluating Model Performance [*]\n")
            self.y_pred = self.model.predict(self.X_val)
            self.accuracy = r2_score(self.y_val, self.y_pred)
            self.m_absolute_error = mean_absolute_error(
                self.y_val, self.y_pred)
            self.rm_squared_error = mean_squared_error(
                self.y_val, self.y_pred, squared=False
            )
            print(
                Fore.CYAN
                + "        Validation R^2 Score is : {:.2f} %".format(
                    self.accuracy * 100
                )
            )
            print(
                Fore.CYAN + "        Validation Mean Absolute Error is :",
                self.m_absolute_error,
            )
            print(
                Fore.CYAN + "        Validation Root Mean Squared Error is :",
                self.rm_squared_error,
            )
            self.regressor_name, self.kfold_accuracy = cv(
                self.model,
                self.predictor,
                self.X_train,
                self.y_train,
                self.cv_folds,
                isReg=True,
                verbose=self.verbose,
                fit_params=self.fit_params,
            )
        except Exception as error:
            print(Fore.RED + "Model Evaluation Failed with error: ", error, "\n")
        finally:
            print(Fore.GREEN + "Model Evaluation Completed [", "\u2713", "]\n")

        if not self.predictor == "nb" and self.tune:
            self.__tuner()

        print(Fore.GREEN + "Completed ANAI Run [", "\u2713", "]\n")
        self.end = time.time()
        final_time = self.end - self.start
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")

    def __fitall(self):
        print(Fore.YELLOW + "Training ANAI [*]\n")
        if self.params != {}:
            warnings.warn(params_use_warning, UserWarning)
            self.params = {}
        for _, self.predictor in enumerate(self.model_to_predict):
            if not self.predictor in self.exclude_models:
                try:
                    (
                        self.model,
                        self.objective,
                    ) = regression_predictor(
                        self.predictor,
                        self.params,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        self.random_state,
                        self.metric,
                        mode="multi",
                        verbose=self.verbose,
                    )
                except Exception as error:
                    print(traceback.format_exc())
                    print(
                        Fore.RED + classifiers[self.predictor],
                        "Model Train Failed with error: ",
                        error,
                        "\n",
                    )
                try:
                    if hasattr(self.model.fit, "verbose"):
                        self.model.fit(
                            self.X_train,
                            self.y_train,
                            verbose=0 if not self.verbose else 1,
                        )
                    else:
                        self.model.fit(self.X_train, self.y_train)
                    self.fit_params = {}
                except Exception as error:
                    print(
                        Fore.RED + regressors[self.predictor],
                        "Model Train Failed with error: ",
                        error,
                        "\n",
                    )
                try:
                    self.y_pred = self.model.predict(self.X_val)
                    self.accuracy = r2_score(self.y_val, self.y_pred)
                    self.m_absolute_error = mean_absolute_error(
                        self.y_val, self.y_pred)
                    self.rm_squared_error = mean_squared_error(
                        self.y_val, self.y_pred, squared=False
                    )
                    self.acc.append(
                        self.accuracy * 100
                    ) if self.accuracy is not None else self.acc.append("NA")
                    self.rmse.append(
                        self.rm_squared_error
                    ) if self.rm_squared_error is not None else self.rmse.append("NA")
                    self.mae.append(
                        self.m_absolute_error
                    ) if self.m_absolute_error is not None else self.mae.append("NA")
                    self.regressor_name, self.kfold_accuracy = cv(
                        self.model,
                        self.predictor,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        all_mode=True,
                        isReg=True,
                        verbose=self.verbose,
                        fit_params=self.fit_params,
                    )
                    self.k_fold_accuracy.append(
                        self.kfold_accuracy
                    ) if self.kfold_accuracy is not None else self.k_fold_accuracy.append(
                        "NA"
                    )
                    self.regressor_model.append(
                        self.model
                    ) if self.model is not None else self.regressor_model.append("NA")
                except Exception as error:
                    print(
                        Fore.RED + regressors[self.predictor],
                        "Evaluation Failed with error: ",
                        error,
                        "\n",
                    )
                finally:
                    self.accuracy = None
                    self.rm_squared_error = None
                    self.m_absolute_error = None
                    self.m_absolute_error = None
                    self.kfold_accuracy = None
                    self.model = None

                if self.tune:
                    self.__tuner(all_mode=True, single_mode=False)
                if self.predictor == "nb":
                    self.best_params = ""
                    self.best_accuracy = self.kfold_accuracy
        self.result_df["R^2 Score"] = self.acc
        self.result_df["Mean Absolute Error"] = self.mae
        self.result_df["Root Mean Squared Error"] = self.rmse
        self.result_df["Cross Validated Accuracy"] = self.k_fold_accuracy
        self.result_df["Model"] = self.regressor_model
        self.result_df = self.result_df.sort_values(
            by=['Cross Validated Accuracy'], ascending=False)
        top_result = self.result_df.sort_values(
            by=['Cross Validated Accuracy'], ascending=False).head(5)
        if self.ensemble:
            estimators = []
            est1 = []
            for i in range(top_result.shape[0]):
                if not top_result.iloc[i]['Name'] == 'K-Nearest Neighbors':
                    estimators.append(
                        top_result.iloc[i]['Model'])
                    est1.append(
                        (top_result.iloc[i]['Name'], top_result.iloc[i]['Model']))
            self.estimators = estimators
            print(Fore.YELLOW + "Ensembling on top {} models\n".format(
                5 if len(estimators) > 5 else len(estimators)))
            try:
                ens_result = self.ensembler.ensemble(
                    self.X_train, self.y_train, self.X_val, self.y_val, cv_folds=self.cv_folds, estimators=estimators, est=est1)
                self.result_df = pd.concat(
                    [self.result_df, ens_result], axis=0)
            except Exception as error:
                print(
                    Fore.RED + "Ensembling Failed with error: ",
                    error,
                    "\n",
                )
        self.result_df = self.result_df.sort_values(
            by=['Cross Validated Accuracy'], ascending=False).reset_index(drop=True)
        print(Fore.GREEN + "Training Done [", "\u2713", "]\n")
        print(Fore.CYAN + "Results Below\n")
        if self.tune:
            self.result_df["Best Parameters"] = self.best_params
            self.result_df["Best Accuracy"] = self.bestacc
            self.result_df["Trained Model"] = self.tuned_trained_model
            self.best_regressor = Best(
                self.result_df.loc[self.result_df["Best Accuracy"].idxmax()],
                self.tune,
                isReg=True,
            )
        else:
            self.best_regressor = Best(
                self.result_df.loc[self.result_df["Cross Validated Accuracy"].idxmax(
                )],
                self.tune,
                isReg=True,
            )
        display(self.result_df.drop(['Model'], axis=1))
        print(Fore.GREEN + "\nCompleted ANAI Run [", "\u2713", "]\n")
        if len(self.model_to_predict) > 1:
            self.model = self.best_regressor.model
            self.end = time.time()
            final_time = self.end - self.start
            self.meta_path = self.save(
                best=True,
            )
            print(
                Fore.CYAN
                + "Saved Best Model at {} ".format(
                    self.meta_path
                ),
                "\n",
            )
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")
        return

    def __tuner(self, all_mode=False, single_mode=True):
        if not all_mode or single_mode:
            print(Fore.YELLOW + "Tuning Started [*]\n")
        if not self.predictor == "nb":
            (
                best_params,
                self.best_accuracy,
                self.best_trained_model,
            ) = anai_tuner(
                self.predictor,
                self.objective,
                self.n_trials,
                self.sampler,
                self.direction,
                self.X_train,
                self.y_train,
                self.cv_folds,
                self.random_state,
                self.metric,
                all_mode=all_mode,
                isReg=True,
            )
        if self.predictor == "nb":
            self.best_params = "Not Applicable"
            self.best_accuracy = 0
        self.best_params.append(self.best_params)
        self.bestacc.append(self.best_accuracy * 100)
        self.tuned_trained_model.append(self.best_trained_model)
        if not all_mode or single_mode:
            print(Fore.GREEN + "Tuning Done [", "\u2713", "]\n")

    def result(self):
        """[Makes a dictionary containing Regressor Name, K-Fold CV Accuracy, RMSE, Prediction set.]

        Returns:

            [dict]: [Dictionary containing :
                        - "Regressor" - Regressor Name
                        - "Accuracy" - Cross Validated CV Accuracy
                        - "RMSE" - Root Mean Square
                        - "YPred" - Array for Prediction set
                        ]
            [dataframe] : [Dataset containing accuracy and best_params
                            for all predictors only when predictor = 'all' is used
                            ]
        """
        if not self.pred_mode == "all":
            self.reg_result["Regressor"] = self.regressor_name
            self.reg_result["Accuracy"] = self.kfold_accuracy
            self.reg_result["RMSE"] = self.rm_squared_error
            self.reg_result["YPred"] = self.y_pred
            reg_result = pd.DataFrame.from_dict(
                self.reg_result, orient='index', columns=['Summary'])
            return reg_result
        if self.pred_mode == "all":
            return self.result_df

    def predict(self, X_test):
        """[Takes test set and returns predictions for that test set]

        Args:
            X_test ([Array]): [Array Containing Test Set]

        Returns:
            [Array]: [Predicted set for given test set]
        """
        if self.pred_mode == "all":
            regressor = copy.deepcopy(self.best_regressor.model)
            print(Fore.YELLOW + "Predicting on Test Set using Best Model[*]\n")
        else:
            regressor = copy.deepcopy(
                self.model
            )
        X_test = np.array(X_test) if type(X_test) == list else X_test
        if isinstance(X_test, pandas.core.frame.DataFrame) or isinstance(X_test, modin.pandas.DataFrame):
            if self.target in X_test.columns:
                X_test = X_test.drop(self.target, axis=1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        X_test = pd.DataFrame(X_test, columns=self.features.columns) if type(
            X_test) == np.ndarray else X_test
        if isinstance(X_test, modin.pandas.DataFrame):
            X_test = X_test._to_pandas()
        y_test = regressor.predict(
            self.sc.transform(self.encoder.transform(X_test)))
        print(Fore.BLUE + "Predicted Value : ", y_test, "\n")
        print(Fore.GREEN + "Prediction Done [", "\u2713", "]\n")
        return y_test

    def save(self, path=None, best=False, **kwargs):
        """
        Saves the model and its scaler to a file provided with a path.
        If no path is provided will create a directory named
        anai_info/models/ and anai_info/scaler/ in current working directory

        Args:

            path ([list]): [List containing path to save the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:

            Path to the saved model and its scaler.
        """
        if not type(path) == list and path != None:
            raise TypeError("Path must be a list")
        if self.pred_mode == "all" and best == False:
            raise TypeError("Cannot save model for all predictors")
        dir_path_model = path[0] if path else "anai_info/models/regression/"
        dir_path_scaler = path[1] if path else "anai_info/scalers/regression/"
        model_name = regressors[self.predictor].replace(" ", "_")
        if best:
            dir_path_model = "anai_info/best/regression/models/"
            dir_path_scaler = "anai_info/best/regression/scalers/"
            model_name = self.best_regressor.name.replace(" ", "_")
        os.makedirs(dir_path_model, exist_ok=True)
        os.makedirs(dir_path_scaler, exist_ok=True)
        timestamp = str(int(time.time()))
        path_model = dir_path_model + model_name + "_" + timestamp
        path_scaler = (
            dir_path_scaler + model_name + "_" + "Scaler" + "_" + timestamp + ".pkl"
        )
        if (
            not kwargs.get("model")
            and not kwargs.get("best")
            and not kwargs.get("scaler")
        ):
            dump(
                self.model, open(path_model + ".pkl", "wb")
            )
            dump(self.sc, open(path_scaler, "wb"))
        else:
            dump(
                kwargs.get("model"), open(path_model + ".pkl", "wb")
            )
            path_model = (
                path_model + ".pkl"
            )
            dump(kwargs.get("scaler"), open(path_scaler, "wb"))
        if not best:
            print("Model Saved at {} and Scaler at {}".format(
                path_model, path_scaler))
        return path_model, path_scaler

    def __load(self, path=None):
        """
        Loads model and scaler from the specified path

        Args:

            path ([list]): [List containing path to load the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            [Model] : [Loaded model]
            [Scaler] : [Loaded scaler]
        """
        model_path = path[0] if path[0] else None
        scaler_path = path[1] if path[1] else None
        if not ".pkl" in model_path and not model_path == None:
            raise TypeError(
                "[Error] Model Filetype not supported. Please use .pkl type "
            )
        if not ".pkl" in scaler_path and not scaler_path == None:
            raise TypeError(
                "[Error] Scaler Filetype not supported. Please use .pkl type "
            )
        if model_path != None and scaler_path != None:
            model = load(open(model_path, "rb"))
            scaler = load(open(scaler_path, "rb"))
            print(
                Fore.GREEN
                + "[Info] Model and Scaler Loaded from {} and {}".format(
                    model_path, scaler_path
                )
            )
            return model, scaler
        elif model_path != None and scaler_path == None:
            model = load(open(model_path, "rb"))
            print(Fore.GREEN +
                  "[Info] Model Loaded from {}".format(model_path))
            return model
        elif model_path == None and scaler_path != None:
            scaler = load(open(scaler_path, "rb"))
            print(Fore.GREEN +
                  "[Info] Scaler Loaded from {}".format(scaler_path))
            return scaler
        else:
            raise ValueError("No path specified.Please provide actual path\n")

    def explain(self, method):
        """
        Returns the importance features of the dataset
        """
        self.explainer.set_params(
            self.features,
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            self.cv_folds,
            self.fit_params,
        )
        if self.pred_mode == "all":
            regressor = copy.deepcopy(self.best_regressor.model)
            print(Fore.YELLOW + "Explaining Best ANAI model [*]\n")
        else:
            regressor = copy.deepcopy(
                self.model
            )
            print(Fore.YELLOW + "Explaining ANAI [*]\n")

        if self.original_predictor == "all":
            raise TypeError(
                "[Error] This method is only applicable on single predictor"
            )
        elif method == "perm":
            self.explainer.permutation(model=regressor)
        elif method == "shap":
            self.explainer.shap(model=regressor)
        else:
            raise NotImplementedError(
                "Technique not implemented. Please choose from perm, shap"
            )
