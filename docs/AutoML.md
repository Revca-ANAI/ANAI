# ANAI

## AutoML Pipeline

### Initialization

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

### Return

        ai : regression or classification object
            Returns a regression or classification object

## Available Methods

### Result

    Gives the result of the model.

        result = ai.result()

        Returns a dataframe of the summary

### Predict

    Predicts the target column for the given data.

        pred = ai.predict(data)

        returns the predictions

### Save Model

    Saves the model to the given path.

        path_model, path_scaler = ai.save(path = [path_model.pkl, path_scaler.pkl])

        Returns the path to the model and scaler

### Explain

    Explains the model predictions.

        exp = ai.explain(method = 'shap')

        Avaliable Methods:
            shap : SHAP
            perm : Permutation

## Examples

### Load Model

    Loads the model from the given path.
    
        ai = anai.run(path = [path_model.pkl, path_scaler.pkl])

### Hyperparameter Tuning

    Use Tune=True to apply Optuna to find best parameters for model.

        ai = anai.run(filepath="data/iris.csv", target="class", predictor="lr", tune = True)

### All Models

    Use predictor = 'all' to use all the models.

        ai = anai.run(filepath="data/iris.csv", target="class", predictor="all")

### Multiple Models

    Pass a list of models to use to predictor.

        ai = anai.run(filepath="data/iris.csv", target="class", predictor=['lr', 'rf'])

### Principal Component Analysis

    Use pca = True to use PCA on Train and Validation set.

        ai = anai.run(filepath="data/iris.csv", target="class", predictor="lr", pca = True)

### Linear Discriminant Analysis

    Use lda = True to use LDA on Train and Validation set.

        ai = anai.run(filepath="data/iris.csv", target="class", predictor="lr", lda = True)
