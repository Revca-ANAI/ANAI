from anai.utils.wrappers.classification_shell import ClassificationWrapper
from sklearn.linear_model import (PoissonRegressor, GammaRegressor, LassoLarsIC,
                                  LassoLarsCV, Lars, LarsCV, TweedieRegressor,
                                  RANSACRegressor, OrthogonalMatchingPursuitCV,
                                  OrthogonalMatchingPursuit, QuantileRegressor,
                                  TheilSenRegressor)
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis


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


class LarsClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = Lars(**params)
        self.num_classes = num_classes - 1


class LarsCVClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = LarsCV(**params)
        self.num_classes = num_classes - 1


class TweedieClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = TweedieRegressor(**params)
        self.num_classes = num_classes - 1


class GeneralizedLinearClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = GeneralizedLinearRegressor(**params)
        self.num_classes = num_classes - 1


class RANSACClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = RANSACRegressor(**params)
        self.num_classes = num_classes - 1


class OrthogonalMatchingPursuitCVClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = OrthogonalMatchingPursuitCV(**params)
        self.num_classes = num_classes - 1


class OrthogonalMatchingPursuitClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = OrthogonalMatchingPursuit(**params)
        self.num_classes = num_classes - 1


class AFMClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = ComponentwiseGradientBoostingSurvivalAnalysis(**params)
        self.num_classes = num_classes - 1


class QuantileClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = QuantileRegressor(**params)
        self.num_classes = num_classes - 1


class TheilSenClassifier(ClassificationWrapper):
    def __init__(self,num_classes=2, **params):
        self.model = TheilSenRegressor(**params)
        self.num_classes = num_classes - 1
