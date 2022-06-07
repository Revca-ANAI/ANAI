import shap
import numpy as np


def shap_feature_importance(columns, X_train, model, *args, **kwargs):
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(X_train))
    if shap_values.ndim == 3:
        shap_values = np.array(
            shap_values[1] + shap_values[2],
        )
    columns = columns
    shap.summary_plot(
        shap_values, X_train, feature_names=columns, plot_type="bar", *args, **kwargs
    )
    shap.summary_plot(shap_values, X_train, feature_names=columns, *args, **kwargs)
    for i in range(len(columns)):
        shap.dependence_plot(
            i, shap_values, X_train, feature_names=columns, *args, **kwargs
        )
