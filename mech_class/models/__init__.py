"""Model wrappers for mech-class classifiers."""

from mech_class.models.composite_head import CompositeHead
from mech_class.models.lightgbm_clf import LightGBMClassifier

__all__ = ["LightGBMClassifier", "CompositeHead"]
