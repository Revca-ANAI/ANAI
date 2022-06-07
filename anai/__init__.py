import os

os.environ["MODIN_ENGINE"] = "dask"

import shutil

if os.path.exists(os.getcwd() + "/dask-worker-space"):
    shutil.rmtree(os.getcwd() + "/dask-worker-space")

import inspect

import warnings

warnings.filterwarnings("ignore")
from colorama import Fore

from optuna.samplers._tpe.sampler import TPESampler

from anai import *
from anai.supervised import Classification, Regression
from anai.utils.connectors import load_data_from_config
from anai.utils.connectors.data_handler import df_loader


def run(
    df=None,
    target: str = None,
    filepath: str = None,
    config: bool = False,
    except_columns: list = [],
    predictor: list = [],
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
    metric: str = None,
    suppress_task_detection=False,
    task=None,
):
    try:
        if not suppress_task_detection:
            print(
                Fore.YELLOW
                + "ANAITaskWarning: Task is getting detected automatically. To suppress this behaviour, set suppress_task_detection=True and specify task with task argument",
                Fore.RESET,
            )
        else:
            if task is None:
                raise ValueError(
                    "task argument is required when suppress_task_detection=True"
                )
        if config:
            if os.path.exists(os.getcwd() + "/anai_config.yaml"):
                df, target = load_data_from_config(
                    os.getcwd() + "/anai_config.yaml", suppress=True
                )
            else:
                raise FileNotFoundError("ANAI Config File Not Found")
        if df is None:
            if filepath is not None:
                df = df_loader(filepath, suppress=True)
            else:
                raise ValueError("Please provide a dataframe or a filepath")
        if __task(df, target) and not suppress_task_detection or task == "regression":
            print(Fore.BLUE + "Task: Regression", Fore.RESET)
            if predictor is None:
                predictor = ["lin"]
            if metric is None:
                metric = "r2"
            regressor = Regression(
                df=df,
                target=target,
                filepath=filepath,
                config=config,
                except_columns=except_columns,
                predictor=predictor,
                params=params,
                tune=tune,
                test_size=test_size,
                cv_folds=cv_folds,
                random_state=random_state,
                pca_kernel=pca_kernel,
                n_components_lda=n_components_lda,
                lda=lda,
                pca=pca,
                n_components_pca=n_components_pca,
                smote=smote,
                k_neighbors=k_neighbors,
                verbose=verbose,
                exclude_models=exclude_models,
                path=path,
                optuna_sampler=optuna_sampler,
                optuna_direction=optuna_direction,
                optuna_n_trials=optuna_n_trials,
                metric=metric,
            )
            return regressor
        elif (
            not __task(df, target)
            and not suppress_task_detection
            or task == "classification"
        ):
            print(Fore.BLUE + "Task: Classification", Fore.RESET)
            if predictor is None:
                predictor = ["lr"]
            if metric is None:
                metric = "accuracy"
            classifier = Classification(
                df=df,
                target=target,
                filepath=filepath,
                config=config,
                except_columns=except_columns,
                predictor=predictor,
                params=params,
                tune=tune,
                test_size=test_size,
                cv_folds=cv_folds,
                random_state=random_state,
                pca_kernel=pca_kernel,
                n_components_lda=n_components_lda,
                lda=lda,
                pca=pca,
                n_components_pca=n_components_pca,
                smote=smote,
                k_neighbors=k_neighbors,
                verbose=verbose,
            )
            return classifier
    except KeyboardInterrupt:
        if os.path.exists(os.getcwd() + "/dask-worker-space"):
            shutil.rmtree(os.getcwd() + "/dask-worker-space")
    except InterruptedError:
        if os.path.exists(os.getcwd() + "/dask-worker-space"):
            shutil.rmtree(os.getcwd() + "/dask-worker-space")


def __get_args(df, target):
    if target not in df.columns:
        raise ValueError(f"target {target} not in df columns")
    target = df[target]
    if target.nunique() > len(df) / 10:
        arg = zip(
            inspect.getfullargspec(Regression).args,
            inspect.getfullargspec(Regression).defaults,
        )
        print("regression")
    else:
        arg = zip(
            inspect.getfullargspec(Classification).args,
            inspect.getfullargspec(Classification).defaults,
        )
        print("classification")
    args = dict(arg)
    args.pop("self")
    return args


def __task(df, target):
    if target not in df.columns:
        raise ValueError(f"target {target} not in df columns")
    target = df[target]
    if target.nunique() > len(df) / 10:
        return True
    else:
        return False
