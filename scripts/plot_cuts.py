"""
Plot variables before and after cuts

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util


def _plot_df(
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
) -> None:
    """
    Plot and show stuff

    """
    # Plot
    fig, ax = plt.subplots(5, 3, figsize=(15, 9))
    sig_colour, bkg_colour = "b", "r"

    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100
    for col, axis in zip(sig_df, ax.ravel()):
        bkg_quantile = np.quantile(bkg_df[col], quantiles)
        sig_quantile = np.quantile(sig_df[col], quantiles)
        bins = np.linspace(
            min(bkg_quantile[0], sig_quantile[0]),
            max(bkg_quantile[1], sig_quantile[1]),
            n_bins,
        )

        # Plot before cuts
        axis.hist(
            sig_df[col], bins=bins, histtype="step", label="sig", color=sig_colour
        )
        axis.hist(
            bkg_df[col], bins=bins, histtype="step", label="bkg", color=bkg_colour
        )

        # Plot after cuts
        axis.hist(
            sig_df[col][sig_predictions == 1],
            bins=bins,
            alpha=0.5,
            color=sig_colour,
        )
        axis.hist(
            bkg_df[col][bkg_predictions == 1],
            bins=bins,
            alpha=0.5,
            color=bkg_colour,
        )

        axis.set_title(col)

        # Add a * to the plot titles for those not used in training the classifier
        if col not in read_data.training_var_names():
            title = axis.get_title()
            axis.set_title(title + "*")

        axis.legend()

    fig.suptitle("Testing sample")
    fig.tight_layout()
    plt.show()


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

    # Predict which of these are signal and background using our classifier
    clf = util.get_classifier()
    sig_predictions = clf.predict(sig_df[list(read_data.training_var_names())])
    bkg_predictions = clf.predict(bkg_df[list(read_data.training_var_names())])

    # Plot histograms of our variables before/after doing these cuts
    _plot_df(sig_df, bkg_df, sig_predictions, bkg_predictions)


if __name__ == "__main__":
    main()
