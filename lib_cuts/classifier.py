from sklearn.ensemble import RandomForestClassifier


class Classifier(RandomForestClassifier):
    """
    Basically just a wrapper around an sklearn classifier that returns predict_proba
    normalised

    """

    def __init__(self, n_sig: int, n_bkg: int, *args, **kwargs):
        """
        Initialise the base classifier with *args and **kwargs and store the signal/bkg ratio
        so that we can use it later for predicting probabilities

        """
        self._sig_bkg_ratio = n_sig / n_bkg

        super().__init__(self, *args, **kwargs)

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        """
        Probability estimates as returned by base class predict_proba, but scaled such that
        the mean probability is 0 (i.e. removing the effect of an imbalanced training set)

        """
        return super().predict_proba(points) / self._sig_bkg_ratio
