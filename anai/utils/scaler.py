import modin.pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from colorama import Fore


class Scaler:
    def __init__(self) -> None:
        pass

    def standardize(self, df, columns):
        try:
            filtered_data = df.drop(columns, axis=1)
            scaled = pd.DataFrame(
                StandardScaler().fit_transform(df[columns]), columns=columns
            )
            combined_data = pd.concat([filtered_data, scaled], axis=1).reset_index(
                drop=True
            )
            return combined_data, scaled
        except Exception as error:
            print(Fore.RED + "Standard Scaler Failed with error :", error)
            return pd.DataFrame(), pd.DataFrame()

    def normalize(self, df, columns):
        try:
            filtered_data = df.drop(columns, axis=1)
            normalized = pd.DataFrame(
                Normalizer().fit_transform(df[columns]), columns=columns
            )
            combined_data = pd.concat([filtered_data, normalized], axis=1).reset_index(
                drop=True
            )
            return combined_data, normalized
        except Exception as error:
            print(Fore.RED + "Normalizer Failed with error :", error)
            return pd.DataFrame(), pd.DataFrame()

    def robust_scaler(self, df, columns):
        try:
            for column in columns:
                try:
                    df[column] = df[column].apply(lambda x: (x - x.mean()) / x.std())
                except Exception as error:
                    print(Fore.RED + "Robust Scaler Failed with error :", error)
            return df
        except Exception as error:
            print(Fore.RED + "Robust Scaler Failed with error :", error)
            return pd.DataFrame()
