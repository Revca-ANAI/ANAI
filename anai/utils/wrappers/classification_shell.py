import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ClassificationWrapper:
    """
    Parent class to covert regression models into classification
    """

    def __init__(self, model, num_classes=1, **kwargs):
        self.prediction = None
        # self.model = model
        self.model = model(**kwargs)
        self.num_classes = num_classes

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X_train):
        pred = self.model.predict(X_train)
        scaled_pred = MinMaxScaler().fit_transform(pred.view(-1, 1))
        self.prediction = np.around(scaled_pred * self.num_classes, 0)
        return self.prediction

    def score(self, y_test, prediction=None):
        if prediction is None:
            pred = self.prediction

    def set_num_classes(self, num):
        self.num_classes = num

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        print(self.model.set_params(**params))
        return self.model.set_params(**params)
