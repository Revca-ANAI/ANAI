import io
from colorama import Fore
import modin.pandas as pd


def __df_loader_single(
    df_filepath="",
    obj=None,
    objfilepath=None,
    suppress=False,
):
    df = None
    flag = 0
    if obj is None:
        print(Fore.YELLOW + "Loading Data [*]\n") if not suppress else None
        path = df_filepath
    else:
        path = objfilepath
        df_filepath = obj["Body"]
        # print(path)
        flag = 1
    if path.endswith(".csv") or path.startswith("http") or path.startswith("https"):
        df = pd.read_csv(df_filepath)
    elif path.endswith(".xlsx"):
        if flag == 1:
            df = pd.read_excel(io.BytesIO(df_filepath.read()))
        else:
            df = pd.read_excel(df_filepath)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(df_filepath)
    elif path.endswith(".h5"):
        df = pd.read_hdf(df_filepath)
    elif path.endswith(".feather"):
        df = pd.read_feather(df_filepath)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(df_filepath)
    elif path.endswith(".json"):
        df = pd.read_json(df_filepath)
    elif path.endswith(".html"):
        df = pd.read_html(df_filepath)
    elif path.endswith(".stata"):
        df = pd.read_stata(df_filepath)
    elif path.endswith(".sas7bdat"):
        df = pd.read_sas(df_filepath)
    elif path.endswith(".msgpack"):
        df = pd.read_msgpack(df_filepath)
    elif path.endswith(".jsonl"):
        df = pd.read_json(df_filepath, lines=True)
    else:
        raise Exception(
            "File extension not supported. Use .csv, .xlsx, .pkl, .h5, .feather, .parquet, .json, .html, .stata, .sas7bdat, .msgpack , .jsonl OR use df argument"
        )
    if obj is None:
        if df is not None:
            print(
                Fore.GREEN + "Data Loaded Successfully [", "\u2713", "]\n"
            ) if not suppress else None

        else:
            print(
                Fore.RED + "Data Loading Failed [", "\u2717", "]\n"
            ) if not suppress else None
    return df


def df_loader(df_filepath, obj=None, objfilepath=None, suppress=False):
    if type(df_filepath) is str:
        df = __df_loader_single(df_filepath, obj, objfilepath, suppress)
    elif type(df_filepath) is list:
        print(Fore.YELLOW + "Loading Data [*]\n")
        df = pd.concat(
            [
                __df_loader_single(df_filepath[i], obj, objfilepath, True)
                for i in range(len(df_filepath))
            ]
        )
        if df is not None:
            print(
                Fore.GREEN + "Data Loaded Successfully [", "\u2713", "]\n"
            ) if not suppress else None

        else:
            print(
                Fore.RED + "Data Loading Failed [", "\u2717", "]\n"
            ) if not suppress else None
    return df
