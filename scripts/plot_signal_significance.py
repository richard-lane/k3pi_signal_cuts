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
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts import read_data, util, metrics
from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


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
    year, sign, magnetisation = "2018", "dcs", "magdown"
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    # We only want the testing data here
    sig_df = sig_df[~sig_df["train"]]
    bkg_df = bkg_df[~bkg_df["train"]]

    # Throw away data to get a realistic proportion of each
    sig_frac = 0.023  # Got this number from k3pi_signal_cuts/scripts/mass_fit.py
    keep_frac = util.weight(
        np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df)))), sig_frac
    )
    sig_keep = np.random.default_rng().random(len(sig_df)) < keep_frac
    print(f"keeping {np.sum(sig_keep)} of {len(sig_keep)}")

    # Find signal probabilities
    clf = get_clf(year, sign, magnetisation)
    training_var_names = list(training_vars.training_var_names())
    sig_probs = clf.predict_proba(sig_df[training_var_names])[:, 1]
    bkg_probs = clf.predict_proba(bkg_df[training_var_names])[:, 1]

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

    plt.savefig("significance_threshholds.png")
    plt.show()

    # Plot delta M for each threshhold
    plt.clf()
    fig, ax = plt.subplots()
    sig_delta_m = sig_df["D* mass"] - sig_df["D0 mass"]
    bkg_delta_m = bkg_df["D* mass"] - bkg_df["D0 mass"]

    hist_kw = {"bins": np.linspace(130, 155), "histtype": "step"}
    for i, threshhold in enumerate(threshholds[:14]):
        if not i % 3:
            sig_keep = sig_probs > threshhold
            bkg_keep = bkg_probs > threshhold
            ax.hist(
                np.concatenate((sig_delta_m[sig_keep], bkg_delta_m[bkg_keep])),
                **hist_kw,
                label=f"{threshhold=:.3f}",
            )

    ax.legend()
    fig.savefig("masses.png")

    plt.show()


if __name__ == "__main__":
    main()
