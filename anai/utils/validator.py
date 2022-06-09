import traceback
from sklearn.model_selection import cross_val_score
import scipy
from anai.utils.configs import *
from colorama import Fore
import pandas as pd
import modin


def pred_check(predictor, pred_type):
    if pred_type == "regression":
        available_predictors = list(regressors_ver.keys())
    elif pred_type == "classification":
        available_predictors = list(classifiers_ver.keys())
    if type(predictor) == str:
        if predictor in available_predictors:
            return True, predictor
        else:
            return False, predictor
    elif type(predictor) == list:
        for i in predictor:
            if i not in available_predictors:
                return False, i
        return True, None


def sparse_check(features, labels):
    features = features
    labels = labels
    ori_feat = features
    ori_labels = labels
    """
        Takes features and labels as input and checks if any of those is sparse csr_matrix.
        """
    try:
        if scipy.sparse.issparse(features[()]):
            features = features[()].toarray()
        elif scipy.sparse.issparse(labels[()]):
            labels = labels[()].toarray()
        return (features, labels)
    except Exception as error:
        pass

    return (ori_feat, ori_labels)


def cv(
    model,
    predictor,
    X_train,
    y_train,
    cv_folds,
    isReg=False,
    all_mode=False,
    verbose=0,
    fit_params={},
    ensembler_name=None,
    **kwargs
):
    """
    Takes predictor, input_units, epochs, batch_size, X_train, y_train, cv_folds, and accuracy_scores dictionary.
    Performs K-Fold Cross validation and stores result in accuracy_scores dictionary and returns it.
    """
    if verbose == False:
        verbose = 0
    elif verbose == True:
        verbose = 2
    if not isReg:
        name = classifiers
        scoring = "accuracy"
    if isReg:
        name = regressors
        scoring = "r2"
    try:
        accuracies = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv_folds,
            scoring=scoring,
            verbose=verbose,
            fit_params=fit_params,
        )
        if not all_mode:
            if not isReg:
                print(
                    "        Cross Validated Accuracy: {:.2f} %".format(
                        accuracies.mean() * 100
                    )
                )
            if isReg:
                print(
                    "        Cross Validated R^2 Score: {:.2f} %".format(
                        accuracies.mean() * 100
                    )
                )
        model_name = name[predictor] if ensembler_name is None else ensembler_name
        accuracy = accuracies.mean() * 100
        if not all_mode:
            print(
                "        Cross Validated Standard Deviation: {:.2f} %".format(
                    accuracies.std() * 100
                ),
                "\n",
            )
        return (model_name, accuracy)

    except Exception as error:
        print(traceback.format_exc())
        print(Fore.RED + "Cross Validation failed with error: ", error, "\n")


def type_check(df):
    # if not isinstance(features, pd.DataFrame) and not isinstance(features, modin.pandas.dataframe.DataFrame):
    #     raise TypeError("Features must be a pandas DataFrame")
    # if not isinstance(labels, pd.Series) and not isinstance(labels, modin.pandas.series.Series):
    #     raise TypeError("Labels must be a pandas Series")
    if not isinstance(df, pd.DataFrame) and not isinstance(df, modin.pandas.dataframe.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
