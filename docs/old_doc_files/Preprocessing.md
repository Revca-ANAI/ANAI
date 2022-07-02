# ANAI

## Preprocessing

ANAI preprocessing pipeline

## Initialization

    import anai
    from anai.preprocessing import Preprocessor
    df = anai.load("data/bodyPerformance.csv")
    prep = Preprocessor(dataset=df, target="class", except_columns=['weight_kg'])

## Available Preprocessing Methods

### Data Summary

    Gives a summary of the data.

        summary = prep.summary()

        Returns a DataFrame

### Column Statistics

    Gives a column wise statistics of the dataset.

        column_stats = prep.column_summary()

        Returns a DataFrame

### Imputing Missing Values

    Imputes the missing values using the statistical methods.

        df1 = prep.impute()

        Returns a imputed DataFrame

### Encoding Categorical Variables

    Encodes the categorical variables.

        df = prep.encode(split = False)

        Returns a encoded DataFrame if split is False else returns a tuple of encoded Features and encoded Labels

### Scaling Variables

    Scales the variables using StandardScaler.

        df = prep.scale(columns = [<List of Columns>], method = 'standard')

        Available methods are 'standardize' and 'normal'

        Returns a scaled DataFrame

### Skewness Correction

    Normalizes the skewness of the data.

        df = prep.skewcorrect()

        Returns a BoxCox Normalised DataFrame.

### Prepare

    Prepares the data for modelling.

        X_train, X_val, y_train, y_val, scaler = prep.prepare(features, labels, test_size, random_state, smote, k_neighbors)

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
