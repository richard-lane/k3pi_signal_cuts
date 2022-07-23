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


def _plot_df(sig_df: pd.DataFrame, bkg_df: pd.DataFrame) -> None:
    """
    Plot and show stuff

    """
    test_sig_df = sig_df[~sig_df["train"]]
    test_bkg_df = bkg_df[~bkg_df["train"]]

    # Plot
    fig, ax = plt.subplots(5, 3, figsize=(15, 9))

    kw = {"histtype": "step"}
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100
    for col, axis in zip(test_sig_df, ax.ravel()):
        bkg_quantile = np.quantile(bkg_df[col], quantiles)
        sig_quantile = np.quantile(sig_df[col], quantiles)
        bins = np.linspace(
            min(bkg_quantile[0], sig_quantile[0]),
            max(bkg_quantile[1], sig_quantile[1]),
            n_bins,
        )

        axis.hist(test_sig_df[col], bins=bins, **kw, label="sig")
        axis.hist(test_bkg_df[col], bins=bins, **kw, label="bkg")
        axis.set_title(col)

        # Add a * to the plots not used in training the classifier
        if col not in read_data.training_var_names():
            title = axis.get_title()
            axis.set_title(title + "*")

        axis.legend()

    fig.suptitle("Test sample")
    fig.tight_layout()
    plt.show()


def main():
    """
    Show plots before and after applying cuts with the classifier

    """
    # Read dataframes of stuff
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    _plot_df(sig_df, bkg_df)

    # Now make the plots but remove the bkg
    clf = util.get_classifier()
    sig_labels = clf.predict(sig_df[list(read_data.training_var_names())])
    bkg_labels = clf.predict(bkg_df[list(read_data.training_var_names())])

    _plot_df(sig_df[sig_labels == 1], bkg_df[bkg_labels == 1])


if __name__ == "__main__":
    main()
