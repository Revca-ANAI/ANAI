from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import numpy as np
import modin.pandas as pd


def permutational_feature_importance(columns, X_test, y_test, model, isReg=False, show_graph=True):
    perm_importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error" if isReg else "accuracy",
        n_repeats=10,
        random_state=42,
    )
    sorted_idx = perm_importance.importances_mean.argsort()
    perm_dict = dict(
        zip(columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    )
    a = dict(sorted(perm_dict.items(), key=lambda item: item[1], reverse=False))
    df1 = pd.DataFrame(a.items(), columns=["Column Name", "Permutation Value"])
    df1["Color"] = np.where(df1["Permutation Value"] < 0, "red", "green")
    if show_graph:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Net",
                x=df1["Permutation Value"],
                y=df1["Column Name"],
                marker_color=df1["Color"],
                orientation="h",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title_text="Permutation Feature Importance",
            xaxis_title="Permutation Value",
            yaxis_title="Feature Name",
        )
        fig.show()
    return df1
