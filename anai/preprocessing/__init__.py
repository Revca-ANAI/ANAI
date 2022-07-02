import copy
import logging
import time
import traceback
from collections import Counter

import category_encoders as ce
import matplotlib.pyplot as plt
import modin
import modin.pandas as pd
import numpy as np
import seaborn as sns
from anai.preprocessing import *
from anai.preprocessing.statistics import column_stats_summary, data_stats_summary
from anai.utils.configs import intro
from anai.utils.encoder import Encoder
from anai.utils.scaler import Scaler
from colorama import Fore
from imblearn.over_sampling import SMOTE
from IPython.display import display
from scipy.special import boxcox1p
from scipy.stats import norm, probplot, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessor:
    def __init__(
        self,
        dataset,
        target: str,
        except_columns: list = [],
    ):
        """ Initialize the Preprocessor class.
        Arguments:
            - dataset: pd.DataFrame
                dataset to be used for preprocessing
            - target: str
                Name of the target column
            - except_columns: list 
                List of columns to exclude from the preprocessing
        """
        
        self.__dataset = dataset
        self.__columns = dataset.columns
        self.__except_columns = except_columns
        self.target = target

        self.columns = []
        self.data = []

        self.tables = []
        self.descriptions = {}
        self.isnull = {}
        self.isna = {}

        self.encoder = Encoder()
        self.scaler = Scaler()
        self.features = self.__dataset.drop(self.target, axis=1)
        self.labels = self.__dataset[self.target]

    def prepare(self, features, labels, test_size, random_state, smote, k_neighbors):
        """
        Prepare the data for modelling

            Arguments:
                - features: pd.DataFrame or np.array
                    features to be used for training
                - labels:  pd.Series or np.array
                    labels to be used for training
                - test_size: Size of the test set
                - random_state: Random state for splitting the data
                - smote: Boolean to use SMOTE or not
                - k_neighbors: Number of neighbors to use for SMOTE
        
            Returns:
                - X_train: Training Features
                - X_val: Validation Features
                - y_train: Training Labels
                - y_val: Validation Labels
                - sc:  Scaler Object
        """
        try:
            if smote is True:
                sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
                features, labels = sm.fit_resample(features, labels)
            # Splitting ---------------------------------------------------------------------
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
            # Scaling ---------------------------------------------------------------------
            sc = StandardScaler()
            if not type(X_train) == np.array and not type(X_val) == np.array:
                X_train = sc.fit_transform(X_train)
                X_val = sc.transform(X_val)
            else:
                X_train = sc.fit_transform(X_train.values)
                X_val = sc.transform(X_val.values)
            return (X_train, X_val, y_train, y_val, sc)
        except Exception as error:
            print(traceback.format_exc())
            print(Fore.RED + "Preprocessing Failed with error: ", error, "\n")

    def __plotter(self, name, text, color):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.distplot(
            self.__dataset[name],
            fit=norm,
            color=color,
            label="Skewness: %.2f" % (self.__dataset[name].skew()),
        )
        plt.title(
            name.capitalize()
            + " Distplot for {} {} Skewness Transformation".format(name, text),
            color="black",
        )
        plt.legend()
        plt.subplot(1, 2, 2)

        probplot(self.__dataset[name], plot=plt)
        plt.show()

    def __skewcheck(self):
        numeric_feats = self.__dataset.dtypes[self.__dataset.dtypes != "object"].index
        if not len(self.__except_columns) == 0:
            if len(self.__except_columns) > len(numeric_feats):
                numeric_feats = set(self.__except_columns) - set(numeric_feats)
            else:
                numeric_feats = set(numeric_feats) - set(self.__except_columns)
        skewed_feats = (
            self.__dataset[numeric_feats]
            .apply(lambda x: skew(x.dropna()))
            .sort_values(ascending=False)
        )
        print(Fore.GREEN + "\nSkewness in numerical features: \n")
        skewness = pd.DataFrame(skewed_feats, columns=["Skewness"])
        display(skewness)
        skew_dict = dict(skewness["Skewness"])
        skewed_features = skewness.index
        return (skewed_features, skew_dict)

    def skewcorrect(self) -> pd.DataFrame:
        """
            Plots distplot and probability plot for non-normalized data and after normalizing the provided data.
            Normalizes data using boxcox normalization

        :returns: Scaled Dataset
        :rtype: pd.DataFrame

        """
        try:
            start = time.time()
            print(Fore.MAGENTA + intro, "\n")
            print(Fore.GREEN + "Started ANAI [", "\u2713", "]\n")
            if not isinstance(self.__dataset, pd.DataFrame):
                print(
                    Fore.RED
                    + "TypeError: This Function expects  Pandas Dataframe but {}".format(
                        type(self.__dataset)
                    ),
                    " is given \n",
                )
                end = time.time()
                print(Fore.GREEN + "Elapsed Time: ", end - start, "seconds\n")
                return

            (skewed_features, skew_dict) = self.__skewcheck()
            for column_name in skewed_features:
                lam = 0
                (mu, sigma) = norm.fit(self.__dataset[column_name])
                print(
                    Fore.CYAN
                    + "Skewness Before Transformation for {}: ".format(column_name),
                    self.__dataset[column_name].skew(),
                    "\n",
                )
                print(
                    Fore.CYAN
                    + "Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                        column_name.capitalize(), mu, column_name.capitalize(), sigma
                    ),
                    "\n",
                )
                self.__plotter(column_name, "Before", "lightcoral")
                try:
                    if skew_dict[column_name] > 0.75:
                        lam = 0.15
                    self.__dataset[column_name] = boxcox1p(
                        self.__dataset[column_name], lam
                    )
                    print(
                        Fore.GREEN
                        + "Skewness After Transformation for {}: ".format(column_name),
                        self.__dataset[column_name].skew(),
                        "\n",
                    )
                    (mu, sigma) = norm.fit(self.__dataset[column_name])
                    print(
                        Fore.GREEN
                        + "Mean before Transformation for {} : {}, Standard Deviation before Transformation for {} : {}".format(
                            column_name.capitalize(),
                            mu,
                            column_name.capitalize(),
                            sigma,
                        ),
                        "\n",
                    )
                    self.__plotter(column_name, "After", "orange")
                except Exception as error:
                    print(
                        Fore.RED + "\nPlease check your dataset's column :",
                        column_name,
                        "Raised Error: ",
                        error,
                        "\n",
                    )
                    pass
            end = time.time()
            print(Fore.GREEN + "Elapsed Time: ", end - start, "seconds\n")
            return self.__dataset

        except Exception as error:
            print(Fore.RED + "Skewness Correction Failed with error : ", error, "\n")

    def impute(self, method: str):
        """Imputes the missing values using the statistical methods.

        Args:
            method (str): Method to be used for imputation.
                Possible values are:
                    - 'mean': Mean imputation
                    - 'median': Median imputation
                    - 'mode': Mode imputation
                    - 'drop': Drop the row if any of the columns has missing values
                    - 'drop_col': Drop the column if any of the rows has missing values
        Raises:
            ValueError: If the method is not one of the above mentioned.

        Returns:
            dataset: pd.DataFrame
                Dataset with imputed values.
        """
        
        
        if method == "mean":
            dataset = self.__dataset.fillna(self.__dataset.mean())
            return dataset
        elif method == "median":
            dataset = self.__dataset.fillna(self.__dataset.median())
            return dataset
        elif method == "mode":
            dataset = self.__dataset.fillna(self.__dataset.mode())
            return dataset
        elif method == "drop":
            dataset = self.__dataset.dropna()
            return dataset
        elif method == "drop_col":
            dataset = self.__dataset.dropna(axis=1)
            return dataset
        else:
            raise ValueError("Invalid Imputing method")

    def summary(self):
        """Prints the summary of the dataset.

        Returns:
            pd.DataFrame: Dataset summary.
        """
        
        stats_summary = data_stats_summary(self.__dataset)
        summary_df = pd.DataFrame.from_dict(
            stats_summary, columns=["Stats"], orient="index"
        )
        return summary_df

    def column_summary(self):
        """Prints the summary of the dataset.
        
        Returns:
            pd.DataFrame: Column summary.
        """
        col_stats = {}
        col_stats_ar = {}
        for i in range(len(self.__dataset.columns)):
            col_stats[self.__dataset.columns[i]] = column_stats_summary(
                self.__dataset, self.__dataset.columns[i]
            )
        for i, j in col_stats.items():
            col_stats_ar[i] = col_stats[i][i]
        col_stats_cp = copy.deepcopy(col_stats_ar)
        for i, j in col_stats_ar.items():
            for k, l in j.items():
                if k == "advanced_stats":
                    for m, n in l.items():
                        col_stats_cp[i][m] = n
        for i, j in col_stats_cp.items():
            if 'advanced_stats' in col_stats_cp[i].keys():
                col_stats_cp[i].pop("advanced_stats")
        col_stats_df = pd.DataFrame.from_dict(col_stats_cp, orient="columns")
        return col_stats_df

    def encode(self, type=None, split = False, **kwargs):
        """ Encodes the categorical variables.
        
        Arguments:
            type {str} -- Type of encoding to be used.
            split {bool} -- If True, splits the dataset into train and test.
            
        Returns:
            pd.DataFrame -- Encoded dataset.
        """
        
        if type == "onehot":
            combined_data, encoded_data = Encoder.one_hot_encoder(self.__dataset)
            return combined_data
        elif type == "label":
            labels = Encoder.label_encoder(self.__dataset)
            return labels
        elif type == "anai":
            features = kwargs["features"]
            labels = kwargs["labels"]
            try:
                cat_features = [
                    i for i in features.columns if features.dtypes[i] == "object"
                ]
                le = None
                if labels.dtype == "O":
                    le = LabelEncoder()
                    labels = le.fit_transform(labels)
                cbe_encoder = ce.cat_boost.CatBoostEncoder(
                    random_state=42, cols=cat_features
                )
                if isinstance(features, modin.pandas.DataFrame):
                    features = features._to_pandas()
                if isinstance(labels, modin.pandas.Series):
                    labels = labels._to_pandas()
                features = cbe_encoder.fit_transform(features, labels)
                return (pd.DataFrame(features), labels, features.columns, cbe_encoder, le)

            except Exception as error:
                print(traceback.format_exc())
                print(Fore.RED + "Encoding Failed with error :", error)
        elif type is None:
            try:
                cat_features = [
                    i
                    for i in self.features.columns
                    if self.features.dtypes[i] == "object"
                ]
                if self.labels.dtype == "O":
                    le = LabelEncoder()
                    self.labels = le.fit_transform(self.labels)
                cbe_encoder = ce.cat_boost.CatBoostEncoder(
                    random_state=42, cols=cat_features
                )
                if isinstance(self.features, modin.pandas.DataFrame):
                    self.features = self.features._to_pandas()
                if isinstance(self.labels, modin.pandas.Series):
                    self.labels = self.labels._to_pandas()
                self.features = cbe_encoder.fit_transform(self.features, self.labels)
                self.features = pd.DataFrame(self.features)
                if split:
                    return (self.features, self.labels)
                else:
                    df = pd.concat([self.features, self.labels], axis=1)
                    return df
            except Exception as error:
                print(traceback.format_exc())
                print(Fore.RED + "Encoding Failed with error :", error)

    def scale(self, columns, method):
        """ Scales the columns.
        
        Arguments:
            type {str} -- Type of scaling to be used.
                    Available methods:
                        - 'standardize': Standard scaling
                        - 'normalize': Normalization
            split {bool} -- If True, splits the dataset into train and test.
            
        Returns:
            pd.DataFrame -- Scaled dataset.
        """
        
        __dataset1 = self.__dataset
        if method == "standardize":
            combined_data, _ = self.scaler.standardize(
                __dataset1,
                columns,
            )
            return combined_data
        elif method == "normalize":
            combined_data, _ = self.scaler.normalize(
                self.__dataset, columns
            )
            return combined_data
        else:
            raise Exception("Invalid Scaler Type")

    def describe(self):
        """Describes the dataset.
        
        Returns:
            pd.DataFrame: Dataset summary.
        """
        self.__dataset.dataset.describe().T.style.bar(
            subset=['mean'],
            color='#606ff2').background_gradient(
            subset=['std'], cmap='PuBu').background_gradient(subset=['50%'], cmap='PuBu')
