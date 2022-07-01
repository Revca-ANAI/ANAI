import six
import sys
sys.modules['sklearn.externals.six'] = six

import inspect
import os
import shutil
import warnings

import modin.pandas as pd
from colorama import Fore
from optuna.samplers._tpe.sampler import TPESampler

from anai import *
from anai.supervised import Classification, Regression
from anai.utils.connectors import load_data_from_config
from anai.utils.connectors.data_handler import __df_loader_single, df_loader

os.environ["MODIN_ENGINE"] = "dask"


if os.path.exists(os.getcwd() + "/dask-worker-space"):
    shutil.rmtree(os.getcwd() + "/dask-worker-space")


warnings.filterwarnings("ignore")


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
    ensemble: bool =True,
    mode: str = "None",
):
    """Initializes ANAI run.

    Arguments:
            df : Pandas DataFrame
                DataFrame to be used for modelling.
            target : str
                Target Column Name 
            except_columns : list, optional
                List of Columns to be excluded from the dataset
            predictor : list
                        Predicting models to be used
            params : dict
                        dictionary containing parameters for model.
                        Not available when using predictor = all or multiple predictors.
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
            tune : boolean
                    when True Applies Optuna to find best parameters for model
                    Default is False
            optuna_sampler : Function
                Sampler to be used in optuna. Default = TPESampler()
            optuna_direction : str
                Direction of optimization. Default = 'maximize'
                Available Directions:
                    maximize : Maximize
                    minimize : Minimize
            optuna_n_trials : int
                No. of trials for optuna. Default = 100
            metric : str,
                metric to be used for model evaluation. Default = 'r2' for regressor and 'accuracy' for classifier
            suppress_task_detection: Bool 
                Whether to suppress automatic task detection. Default = False
            task : str
                Task to be used for model evaluation. Default = None
                Only applicable when suppress_task_detection = True
                Available Tasks:
                    classification : Classification
                    regression : Regression
            ensemble : bool
                Whether to use ensemble. Default = True
            mode : str
                Model selection mode. Default = 'None'
        Returns:

            ai : regression or classification object
                Returns a regression or classification object

        Example:
            import anai
            
            ai = anai.run(
                        filepath='examples/Folds5x2_pp.xlsx', 
                        target='PE',
                        predictor=['lin'],
            )

    """
    
    
    
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
            if len(predictor) == 0 and mode is None:
                predictor = ["lin"]
            if mode == 'auto':
                predictor = ['all']
                ensemble = True
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
                ensemble=ensemble,
            )
            return regressor
        elif (
            not __task(df, target)
            and not suppress_task_detection
            or task == "classification"
        ):
            print(Fore.BLUE + "Task: Classification", Fore.RESET)
            if len(predictor) == 0 and mode is None:
                predictor = ["lr"]
            if mode == 'auto':
                predictor = ['all']
                ensemble = True
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
                ensemble=ensemble,
                exclude_models=exclude_models,
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


def load(df_filepath):
    """Loads a dataframe from a filepath.

    Args:
        df_filepath (str): Filepath of the dataframe to be loaded.

    Returns:
        pd.DataFrame : Loaded dataframe.
    """
    
    
    suppress = False
    if type(df_filepath) is str:
        df = __df_loader_single(df_filepath, suppress=False)
    elif type(df_filepath) is list:
        print(Fore.YELLOW + "Loading Data [*]\n")
        df = pd.concat(
            [
                __df_loader_single(df_filepath[i], suppress=True)
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
