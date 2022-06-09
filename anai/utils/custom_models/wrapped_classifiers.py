from anai.utils.wrappers.classification_shell import ClassificationWrapper
from sklearn.linear_model import (PoissonRegressor, GammaRegressor, LassoLarsIC,
                                  LassoLarsCV)


class PoissonClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = PoissonRegressor(**params)
        self.num_classes = num_classes - 1


class GammaClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = GammaRegressor(**params)
        self.num_classes = num_classes - 1


class LassoLarsICClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = LassoLarsIC(**params)
        self.num_classes = num_classes - 1


class LassoLarsCVClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = LassoLarsCV(**params)
        self.num_classes = num_classes - 1

