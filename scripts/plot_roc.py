"""
Plot ROC curve for our classifier

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def main():
    """
    ROC curve

    """
    # Read dataframes of stuff
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    combined_df = pd.concat((sig_df, bkg_df))
    combined_y = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    training_labels = list(training_vars.training_var_names())
    predicted_probs = get_clf(year, sign, magnetisation).predict_proba(
        combined_df[training_labels]
    )[:, 1]
    fpr, tpr, _ = roc_curve(combined_y, predicted_probs)
    score = roc_auc_score(combined_y, predicted_probs)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"score={score:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    fig.tight_layout()

    plt.savefig("roc.png")

    plt.show()


if __name__ == "__main__":
    main()
