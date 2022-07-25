"""
Plot the cut efficiency for the testing sample

Ideally it will be constant in decay time

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util


def _plot_ratio(
    ax: plt.Axes,
    numerator: np.ndarray,
    denominator: np.ndarray,
    bins: np.ndarray,
    label: str,
):
    """
    Plot binned ratio of two arrays on an axis

    :param ax: axes to plot on
    :param numerator: the variable we want to plot the efficiency in. Efficiency is numerator/denominator counts in each bin
    :param numerator: the variable we want to plot the efficiency in
    :param bins: bins for variable x
    :param label: axis label for the efficiency

    """
    numerator_indices = np.digitize(numerator, bins)
    denominator_indices = np.digitize(denominator, bins)

    ratio, error = [], []
    n_bins = len(bins) - 1
    for i in range(1, n_bins + 1):
        n_num = np.sum(numerator_indices == i)
        n_denom = np.sum(denominator_indices == i)

        err_num = np.sqrt(n_num)
        err_denom = np.sqrt(n_denom)

        ratio.append(n_num / n_denom)
        error.append(
            ratio[-1] * np.sqrt((err_num / n_num) ** 2 + (err_denom / n_denom) ** 2)
        )

    ratio = np.array(ratio)
    error = np.array(error)

    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2
    ax.errorbar(
        centres,
        ratio,
        xerr=widths,
        yerr=error,
        fmt=".",
        label=label,
    )


def main():
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Predict which of these are signal and background using our classifier
    clf = util.get_classifier()
    sig_predictions = clf.predict(sig_df[list(read_data.training_var_names())])
    bkg_predictions = clf.predict(bkg_df[list(read_data.training_var_names())])

    sig_t = sig_df["time"]
    bkg_t = bkg_df["time"]

    fig, ax = plt.subplots()
    bins = np.linspace(0, 8, 15)
    _plot_ratio(ax, sig_t[sig_predictions == 1], sig_t, bins, "sig")
    _plot_ratio(ax, bkg_t[bkg_predictions == 1], bkg_t, bins, "bkg")

    ax.legend()
    ax.set_xlabel("t / ps")
    ax.set_ylabel("efficiency")

    fig.suptitle("Cut efficiency")

    plt.show()


if __name__ == "__main__":
    main()
