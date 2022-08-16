"""
Get a classifier

"""
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from . import definitions


def classifier(year: str, sign: str, magnetisation: str) -> GradientBoostingClassifier:
    """
    The right classifier

    """
    with open(definitions.classifier_path(year, sign, magnetisation), "rb") as f:
        return pickle.load(f)
