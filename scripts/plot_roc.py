"""
Plot ROC curve for our classifier

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    combined_df = pd.concat((sig_df, bkg_df))
    combined_y = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    predicted_probs = util.get_classifier().predict_proba(
        combined_df[list(read_data.training_var_names())]
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

    plt.show()


if __name__ == "__main__":
    main()
