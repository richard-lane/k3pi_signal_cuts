"""
Get a classifier

"""
import pickle
from sklearn.ensemble import RandomForestClassifier
from . import definitions


def classifier(year: str, sign: str, magnetisation: str) -> RandomForestClassifier:
    """
    The right classifier

    """
    with open(definitions.classifier_path(year, sign, magnetisation), "rb") as f:
        return pickle.load(f)
