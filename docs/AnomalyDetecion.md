# ANAI

## Anomaly Detection

### Usage

    import anai
    
    from anai.unsupervised import anomaly_detector
    
    df = anai.load(filepath='data/iris.csv')

    (anomaly_combined,
        data_with_outliers,
        data_with_Inliers,
        anomaly_summary) = anomaly_detector(dataset = df, target = 'class', model = ['IForest', 'CBLOF'])

### Returns

    anomaly_combined : DataFrame
        df modified with anomaly scores and anomaly
    data_with_outliers : DataFrame
        Only outliers
    data_with_Inliers : DataFrame
        Only Inliers
    anomaly_summary : DataFrame
        Summary of the anomaly detection
