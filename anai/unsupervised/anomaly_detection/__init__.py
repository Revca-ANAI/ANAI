import traceback

import modin
import numpy as np
import pandas as pd
from anai.unsupervised.anomaly_detection.configs import configs
from IPython.display import display


class AnomalyDetector:
    def __init__(
        self,
        outliers_fraction=0.05,
        random_state=42,
        n_neighbors=5,
        filepath="anomaly_models_result.csv",
    ):
        self.anomaly_models, self.anomaly_models_name = configs(
            outliers_fraction=outliers_fraction,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )
        self.anomaly_models_result_file = filepath
        self.data = None
        self.model_name = None
        self.result_df = None
        self.mode = "Single"
        self.df = pd.DataFrame()
        self.inliers_df = pd.DataFrame()
        self.outliers_df = pd.DataFrame()
        self.model_code = None

    def get_availabel_models(self):
        return self.anomaly_models_name

    def __fit(self, data, model_name=None, mode="single", except_models=[]):
        """
        Fits the model with the given data.
            params:
                data: data to be fitted.
                model_name: name of the model to be fit.
                    Get the list of available models by calling get_availabel_models()
                    default: None
                mode: 'single' or 'multi'
                    default: 'single'
                except_models: list of models to be excluded from the fit.
                    default: []
        """
        try:
            if model_name != None:
                self.model_code = model_name
                self.model_name = self.anomaly_models_name[self.model_code]
                self.model = self.anomaly_models[self.model_name]
                # print('hey')
            self.mode = mode
            if mode == "multi":
                self.__fit_all(data, except_models=except_models)
                return

            self.data = data
            self.model.fit(data)
        except Exception as e:
            print(traceback.format_exc())
            print(
                "[{}] Model fit failed with error: {}".format(type(e), e), "\n"
            ) if self.mode == "single" else print("Fit failed with error: {}".format(e))
        # finally:
        #     print(
        #         "{} Model fit completed.".format(
        #             self.model_name if self.model_name != None else 0
        #         )
        #     ) if mode == "single" else None

    def __fit_all(self, data, except_models=[]):
        print("Started fitting all models.")
        self.result_df = pd.DataFrame(
            columns=[
                "Model Name",
                "Total Data",
                "No. of Outliers",
                "No. of Inliers",
                "Model",
            ]
        )
        models_to_be_fit_keys = list(
            set(self.get_availabel_models().keys()) - set(except_models)
        )
        models_to_be_fit = {
            key: self.get_availabel_models()[key] for key in models_to_be_fit_keys
        }
        for i, _ in models_to_be_fit.items():
            self.__fit(data, model_name=i, mode="fit_all")
            self.result_df.loc[i] = self.summary()
            self.result_df.to_csv(self.anomaly_models_result_file, index=False)
        print("All models fit completed.\n")
        display(self.result_df)

    def predict(self, data):
        """
        Predicts the anomaly for the given data.

        params:
            data: data to be predicted.
        """
        if self.mode == "single" or self.mode == "fit_all":
            return self.model.predict(data)
        else:
            print("Predict is not supported in multi mode.")

    def summary(self):
        """
        Returns the summary of the model.
        returns:
            A pandas.DataFrame with the summary of the model.
        """
        if self.mode == "single" or self.mode == "fit_all":
            y_pred = self.predict(self.data)
            n_outliers = np.count_nonzero(y_pred == 1)
            n_inliers = len(y_pred) - np.count_nonzero(y_pred)
            result = {
                # "Model Name": self.model_name,
                "Total Data": len(self.data),
                "No. of Outliers": n_outliers,
                "No. of Inliers": n_inliers,
                # "Model": self.model,
            }
            return pd.DataFrame(result, index=[0]) if self.mode == "single" else result
        else:
            print("Summary is not supported in multi mode.")

    def score(self, data):
        """
        Returns the anomaly score for the given data.

        params:
            data: data to be scored.
        """
        if self.mode == "single":
            return self.model.decision_function(data) * -1

        else:
            print("Score is not supported in multi mode.")

    def update_dataframe(self, data):
        """
        Updates the dataframe with the new data.

        params:
            data: data to be updated.
        """
        data2 = data.copy()
        if self.mode == "single":
            data2.loc[:, "anomaly"] = self.predict(data.values)
            data2.loc[:, "anomaly_score"] = self.score(self.data)
            return data2
        else:
            print("Update dataframe is not supported in multi mode.")

    def validate(self, data, except_columns, model_name, mode):
        if type(except_columns) is not list:
            raise TypeError("except_columns must be a list.")
        if (
            not isinstance(data, pd.DataFrame)
            and not isinstance(data, np.ndarray)
            and not isinstance(data, modin.pandas.dataframe.DataFrame)
        ):
            raise TypeError("data must be a type of pandas.DataFrame or numpy.ndarray.")
        self.model_code = model_name
        self.model_name = self.anomaly_models_name[self.model_code]
        if self.model_name in list(self.anomaly_models.keys()):
            self.model = self.anomaly_models[self.model_name]
        else:
            print(self.model_name, self.model_code)
            raise TypeError(
                "Model not found. Please check available models by using get_available_models()."
            )
        if type(data) is pd.DataFrame or isinstance(
            data, modin.pandas.dataframe.DataFrame
        ):
            if data.shape[0] == 0:
                raise ValueError("data must not be empty.")
            elif data.shape[1] == 0:
                raise ValueError("data must contain atleast two features.")
        if type(data) is np.ndarray:
            if data.shape[0] == 0:
                raise ValueError("data must not be empty.")
            elif data.shape[1] == 0:
                raise ValueError("data must contain atleast two features.")
        if mode not in ["single", "multi"]:
            raise TypeError('mode must be either "single" or "multi".')

    def engine(
        self, data, except_columns=[], model_name=None, mode="single", except_models=[]
    ):
        """
        Fits the model with the given data.
            params:
                data: data to be fitted.
                    type: pandas.DataFrame or np.array
                model_name: name of the model to be fit.
                    default: None
                mode: 'single' or 'multi'
                    default: 'single'
                except_columns: list of columns to be excluded from the fit. (Only used when data is a dataframe)
                    default: []
                outliers_fraction: fraction of outliers to be excluded from the fit.
                    default: 0.05
                except_models: list of models to be excluded from the fit.
                    default: []
            returns:
                result_df: dataframe with anomaly scores. (Only used when mode is 'single' and data is a dataframe)
                outliers_df : dataframe with anomaly scores. (Only used when mode is 'single' and data is a dataframe)
                inlier_df : dataframe without anomaly scores. (Only used when mode is 'single' and data is a dataframe)
                summary: summary of the fit.
                    type: pd.DataFrame
        """
        self.validate(data, except_columns, model_name, mode)
        if type(data) == pd.DataFrame or isinstance(
            data, modin.pandas.dataframe.DataFrame
        ):
            if except_columns != None or except_columns != []:
                X = data.drop(except_columns, axis=1)
                X = X.values
        elif type(data) == np.ndarray:
            X = data.values
        if mode == "single":
            self.__fit(X)
        if mode == "multi":
            self.__fit(X, mode="multi", except_models=except_models)
        summary = self.summary()
        if type(data) == pd.DataFrame:
            if mode == "single":
                self.df = self.update_dataframe(data)
                self.outliers_df = self.df[self.df["anomaly"] == 1]
                self.inliers_df = self.df[self.df["anomaly"] == 0]
        return self.df, self.outliers_df, self.inliers_df, summary
