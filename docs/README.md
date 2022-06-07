# ANAI

## Welcome to ANAI Documentation

### What is ANAI?

ANAI is an Automated Machine Learning Python Library that works with tabular data. It is intended to save time when performing data analysis. It will assist you with everything right from the beginning i.e Ingesting data using the inbuilt connectors, preprocessing, feature engineering, model building, model evaluation, model tuning and much more.

## Getting started

Let's get started.

### Installation

    1) Python venv:
        pip install anai

    2) Anaconda:
        conda install anai

### Available Modelling Techniques

    1) Classification

    2) Regression

### Basic Example

    import anai
    ai = anai.run(filepath="data/iris.csv", target="class", predictor="lr")

### Arguments

        df : Pandas DataFrame
            DataFrame to be used for modelling.
        target : str
            Target Column Name 
        except_columns : list, optional
            List of Columns to be excluded from the dataset
        predictor : list
                    Predicting models to be used
        params : dict
                    dictionary containing parameters for model
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

### Return

        ai : regression or classification object
            Returns a regression or classification object

### ANAI Preprocessing

More about Preprocessing can be found in the [Preprocessing](#preprocessing) section.

### ANAI AutoML

More information about AutoML can be found [AutoML](#anai-automl) section

### Algorithms

More about Algorithms can be found in the [Algorithms](#algorithms) section.

### Anomaly Detection

More about Anomaly Detection can be found in the [Anomaly Detection](#anomaly-detection) section.

### Ingesting data using inbuilt connectors

More about Ingesting data using inbuilt connectors can be found in the [Ingesting data using inbuilt connectors](#ingesting-data-using-inbuilt-connectors) section.
