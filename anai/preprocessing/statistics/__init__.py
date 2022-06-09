import numpy as np
import modin.pandas as pd
from scipy.stats import shapiro
from dateutil.parser import parse
from fuzzywuzzy import fuzz
from anai.unsupervised.anomaly_detection import AnomalyDetector


def column_negatives(dff, col):
    l1 = []
    for i in dff[col]:
        if i < 0:
            l1.append([i])
    return list(l1)


def shapiro_test(df, type):
    if df.dtype != "O":
        wilk, p = shapiro(df)
        if type == "p":
            return p
        if type == "wilk":
            return wilk
    else:
        return "NA"


def is_date(string, fuzzy=False):
    try:
        parse(str(string), fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def dtype(df, col):
    if df[col].dtype == "O":
        if not is_date(df[col].iloc[0]):
            return "Categorical"
        elif is_date(df[col].iloc[0]):
            return "Time Series"
    elif df[col].dtype == "int64" or df[col].dtype == "float64":
        return "Numeric"
    else:
        return "Unknown"


def dtype_ver(df, col):
    if df[col].dtype == "O":
        if not is_date(df[col].iloc[0]):
            return "Categorical", ""
        elif is_date(df[col].iloc[0]):
            return "Categorical", "Time Series"
    elif df[col].dtype == "int64" or df[col].dtype == "float64":
        return "Numeric", ""
    else:
        return "Unknown", ""

def shap(df, col):
    return "{:0.2f}".format(
        float(shapiro(df[col])[0]) if df[col].dtype != "O" else "NA"
    )

def most_frequent_values(df, col):
    return (
        df[col].value_counts()[:1].index.tolist()[0] if df[col].dtype == "O" else "NA"
    )

def column_stats_summary(df, col):
    if "identi" in col.lower():
        return {
            col: {
                "Type Error": "ID column is not allowed",
                "hide": True,
            }
        }

    elif col not in ["id", "ID", "Id", "Unnamed:0"]:
        return {
            col: {
                "Type": dtype(df, col),
                "Missing Value %": str(df[col].isna().sum() * 100 / len(df)),
                "Mean": "{:0.2f}".format(float(df[col].mean()))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "Mode": "{:0.2f}".format(float(df[col].mode().to_list()[0]))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "Maximum value": "{:0.2f}".format(float(df[col].max()))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "Median": "{:0.2f}".format(float(df[col].quantile(0.5)))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "Minimum value": "{:0.2f}".format(float(df[col].min()))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "Standard Deviation": "{:0.2f}".format(float(df[col].std()))
                if dtype_ver(df, col)[0] != "Categorical"
                else "NA as column dtype is Categorical",
                "advanced_stats": {
                    "99% Quartile": "{:0.2f}".format(float(df[col].quantile(0.99)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "90% Quartile": "{:0.2f}".format(float(df[col].quantile(0.9)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "66% Quartile": "{:0.2f}".format(float(df[col].quantile(0.66)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "33% Quartile": "{:0.2f}".format(float(df[col].quantile(0.33)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "10% Quartile": "{:0.2f}".format(float(df[col].quantile(0.1)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "1% Quartile ": "{:0.2f}".format(float(df[col].quantile(0.01)))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Variance": "{:0.2f}".format(float(df.var()[col]))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Monotonic": "{:0.2f}".format(float(df[col].is_monotonic))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Mean Absolute Deviation": "{:0.2f}".format(df[col].mad())
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "No. of Unique Values": len((list(df[col].unique()))),
                    "No. of Negative Values": len(column_negatives(df, col))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Percentage Infinite Values": str(
                        np.isinf(df[col]).values.sum() * 100 / len(df)
                    )
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Skewness": "{:0.2f}".format(float(df[col].skew()))
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Shapiro_W": shap(df, col)
                    if dtype_ver(df, col)[0] != "Categorical"
                    else "NA as column dtype is Categorical",
                    "Most frequent value": most_frequent_values(df, col)
                    if dtype_ver(df, col)[0] == "Categorical"
                    else "NA",
                },
            },
            "hide": True if dtype_ver(df, col)[1] == "Time Series" else False,
        }
    elif col in ["id", "ID", "Id", "Unnamed: 0"]:
        return {
            col: {
                "Type Error": "ID column is not allowed",
                "hide": True,
            }
        }


def data_stats_summary(df):
    anom = AnomalyDetector()
    df2 = df.fillna(df.mean())
    X = []
    for i in df2.columns:
        if dtype_ver(df2, i)[0] == "Numeric":
            X.append(i)
    _, _, _, summary = anom.engine(
        df2[X],
        mode="single",
        model_name="IForest",
    )
    no_of_anomalies = summary["No. of Outliers"][0]
    return {
        "No. of Cells": int(df.size),
        "No. of Variables": int(df.shape[1]),
        "No. of Records": len(df),
        "Missing Cells": str(round((df.isnull().sum().sum() / df.size) * 100, 1))
        + " %",
        "Missing Cells Count": int(df.isnull().sum().sum()),
        "Duplicacy": "{:0.2f} %".format(
            ((len(df) - len(df.drop_duplicates())) / len(df)) * 100
        ),
        "Duplicate Cell Count": int(len(df) - len(df.drop_duplicates())),
        "Anomaly Count": int(no_of_anomalies),
    }
