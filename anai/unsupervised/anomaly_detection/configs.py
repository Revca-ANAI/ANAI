from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF

def configs(outliers_fraction, random_state=42, n_neighbors=35):
    anomaly_models_name = {
        "IForest": "Isolation Forest",
        "CBLOF": "Cluster-based Local Outlier Factor",
    }

    anomaly_models = {
        "Isolation Forest": IForest(
            contamination=outliers_fraction,
            random_state=random_state,
            n_jobs=-1,
            behaviour="new",
        ),
        "Cluster-based Local Outlier Factor": CBLOF(
            contamination=outliers_fraction,
            check_estimator=False,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    return anomaly_models, anomaly_models_name
