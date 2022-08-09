"""
Plot variables before and after cuts

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def _plot(
    axis: plt.Axes,
    signal: np.ndarray,
    bkg: np.ndarray,
    sig_predictions: np.ndarray,
    bkg_predictions: np.ndarray,
) -> None:
    """
    Plot signal + bkg on an axis

    """
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100

    sig_quantile = np.quantile(signal, quantiles)
    bkg_quantile = np.quantile(bkg, quantiles)

    bins = np.linspace(
        min(bkg_quantile[0], sig_quantile[0]),
        max(bkg_quantile[1], sig_quantile[1]),
        n_bins,
    )

    # Plot before cuts
    axis.hist(signal, bins=bins, label="sig", histtype="step", color="b")
    axis.hist(bkg, bins=bins, label="bkg", histtype="step", color="r")

    # Plot after cuts
    axis.hist(
        signal[sig_predictions == 1], bins=bins, label="sig", alpha=0.5, color="b"
    )
    axis.hist(bkg[bkg_predictions == 1], bins=bins, label="bkg", alpha=0.5, color="r")

    axis.legend()


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = get_clf(year, sign, magnetisation)

    training_labels = list(training_vars.training_var_names())
    sig_predictions = clf.predict(sig_df[training_labels])
    bkg_predictions = clf.predict(bkg_df[training_labels])

    # Plot histograms of our variables before/after doing these cuts
    columns = list(training_vars.training_var_names()) + ["D0 mass", "D* mass"]
    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for col, axis in zip(columns, ax.ravel()):
        _plot(axis, sig_df[col], bkg_df[col], sig_predictions, bkg_predictions)

        title = col if col in training_vars.training_var_names() else col + "*"
        axis.set_title(title)

    # Let's also plot the mass difference
    _plot(
        ax.ravel()[-1],
        sig_df["D* mass"] - sig_df["D0 mass"],
        bkg_df["D* mass"] - bkg_df["D0 mass"],
        sig_predictions,
        bkg_predictions,
    )
    ax.ravel()[-1].set_title(r"$\Delta$M*")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
