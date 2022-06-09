from anai.unsupervised.anomaly_detection import AnomalyDetector
import warnings


def anomaly_detector(dataset, target, model=["IForest"]):
    if dataset.isnull().values.any():
        warnings.warn("Dataset contains null values. Imputing with mean.")
        dataset = dataset.fillna(dataset.mean())
    if len(model) == 1:
        mode = "single"
    elif len(model) > 1:
        mode = "multi"
    anom = AnomalyDetector()
    X = dataset.drop([target], axis=1)
    (
        anomaly_combined,
        data_with_outliers,
        data_with_Inliers,
        anomaly_summary,
    ) = anom.engine(
        X,
        mode=mode,
        model_name=model,
        except_models=[],
    )
    return (
        anomaly_combined,
        data_with_outliers,
        data_with_Inliers,
        anomaly_summary,
    )
