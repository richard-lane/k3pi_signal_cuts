"""
Plot signal significance of the testing sample

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, util, metrics


def main():
    """
    We have to make a choice about the threshhold value for predict_proba above which we consider
    an event to be signal. By default in sklearn this is 0.5, but maybe we will find a better signal
    significance by choosing a different value.

    Note that the values returned by predict_proba may not correspond exactly to probabilities.
    This can be checked by running the calibration curve script, but chances are that it's good
    enough.

    Plots signal significances for various values of this threshhold.

    """
    # Read dataframes of stuff
    sig_df = util.read_dataframe(background=False)
    bkg_df = util.read_dataframe(background=True)

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Find signal probabilities
    training_var_names = list(read_data.training_var_names())
    sig_probs = util.get_classifier().predict_proba(sig_df[training_var_names])[:, 1]
    bkg_probs = util.get_classifier().predict_proba(bkg_df[training_var_names])[:, 1]

    # For various values of the threshhold, find the signal significance

    def sig(threshhold: float) -> float:
        n_sig = np.sum(sig_probs > threshhold)
        n_bkg = np.sum(bkg_probs > threshhold)

        return metrics.signal_significance(n_sig, n_bkg)

    x_range = 0.1, 0.95
    threshholds = np.linspace(*x_range, 31)
    significances = [sig(threshhold) for threshhold in threshholds]

    # Find the max of values
    max_index = np.argmax(significances)
    max_response = significances[max_index]
    max_threshhold = threshholds[max_index]

    # Interpolate the threshholds
    lots_of_points = np.linspace(*x_range, 1000)
    response = interp1d(threshholds, significances)(lots_of_points)

    fig, ax = plt.subplots()
    ax.plot(threshholds, significances, "k+")
    ax.plot(lots_of_points, response, "k--")
    ax.plot(max_threshhold, max_response, "ro")

    # Plot an arrow
    length = 10
    plt.arrow(
        max_threshhold,
        max_response - length,
        0,
        length,
        length_includes_head=True,
        color="r",
    )
    plt.text(max_threshhold, max_response - 1.1 * length, f"{max_threshhold=}")

    ax.set_xlabel("probability threshhold")
    ax.set_ylabel("signal significance")
    plt.show()


if __name__ == "__main__":
    main()
