# ANAI

## Explainable ANAI

ANAI model predictions can be explained using SHAP or Permutation.

### Initialize

    import anai
    ai = anai.run(filepath="data/iris.csv", target="class", predictor="lr")

### Explain

    1) Permutation:

        ai.explain(method = 'perm')

![Perm](https://revca-assets.s3.ap-south-1.amazonaws.com/perm.png)

    2) SHAP:

        ai.explain(method = 'shap')

![SHAP](https://revca-assets.s3.ap-south-1.amazonaws.com/shap.png)

### Surrogate Explainer

ANAI uses SHAP tree explainer to explain model predictions. So if the explainer fails to explain certain model it will switch to Surrogate Mode and use Decision Tree Surrogate Model to explain the original trained model. Surrogate mode is available for SHAP and Permutation exaplainers.

    ai = anai.run(filepath="data/iris.csv", target="class", predictor="knn")
    ai.explain(method = 'shap')

As you can see, SHAP explainer failed to explain the model. So it switched to Surrogate Mode and used Decision Tree Surrogate Model to explain the original trained model.

![Surrogate](https://revca-assets.s3.ap-south-1.amazonaws.com/surrogate_shap.png)
