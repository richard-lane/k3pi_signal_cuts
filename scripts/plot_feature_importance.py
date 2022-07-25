"""
Plot feature importance for the classifier

"""
import sys
import pathlib
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import util, read_data


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Predict which of these are signal and background using our classifier
    clf = util.get_classifier()

    fig, ax = plt.subplots(figsize=(10, 5))

    importances = clf.feature_importances_
    ax.bar(range(len(importances)), importances)
    ax.set_xticks(range(len(importances)), read_data.training_var_names(), rotation=45)

    fig.suptitle("Feature Importances")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
