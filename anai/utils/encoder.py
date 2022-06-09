import traceback
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from colorama import Fore
import pandas as pd


class Encoder:
    def __init__(self) -> None:
        pass

    def one_hot_encoder(database):
        try:
            combined_data = pd.DataFrame()
            encoded_data = pd.DataFrame()
            cat_columns = [
                i for i in database.columns if database.dtypes[i] == "object"
            ]
            for i in cat_columns:
                if "date" in i.lower():
                    cat_columns.remove(i)
            if len(cat_columns) >= 1:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoder.fit(database[cat_columns])
                filter_data = database.drop(cat_columns, axis=1)
                encoded_data = pd.DataFrame(
                    encoder.transform(database[cat_columns]),
                    columns=encoder.get_feature_names_out(cat_columns),
                )
                combined_data = pd.concat([filter_data, encoded_data], axis=1).reindex()
                return combined_data, encoded_data
            else:
                return database, None
        except Exception as error:
            print(traceback.format_exc())
            print(Fore.RED + "One Hot Encoding Failed with error :", error)

    def label_encoder(database):
        # for labels in columns:

        try:
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            return labels
        except Exception as error:
            return database
            print(Fore.RED + "Label Encoding Failed with error :", error)
