"""
Do a mass fit to the real data (before BDT cuts) to estimate the amount of signal
and background that we expect in our WS sample

This requires you/me to have cloned the k3pi_mass_fit repo in the same dir as
you cloned k3pi_signal_cuts; also requires some of the real data dataframes
to exist

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_mass_fit"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from libFit import pdfs
from libFit.fit import simultaneous_fit
from lib_data import get


def _delta_m(data: pd.DataFrame) -> np.ndarray:
    """ D* - D0 Mass """
    return data["D* mass"] - data["D0 mass"]


def _plot_fit(
    rs: np.ndarray, ws: np.ndarray, params: tuple
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a fit, assuming we used time bin number 5

    """
    bins = np.linspace(140, 152, 100)
    centres = (bins[1:] + bins[:-1]) / 2
    hist_kw = {"density": True, "histtype": "step"}
    fig, ax = plt.subplots(1, 2)

    rs_params = params[:-1]
    ws_params = (params[-1], *params[1:-1])

    rs_counts, _ = np.histogram(rs, bins)
    ws_counts, _ = np.histogram(ws, bins)

    rs_err = np.sqrt(rs_counts)
    ws_err = np.sqrt(ws_counts)

    rs_scale = len(rs) * (bins[1] - bins[0])
    ws_scale = len(ws) * (bins[1] - bins[0])

    ax[0].errorbar(centres, rs_counts, yerr=rs_err, fmt="k.")
    ax[1].errorbar(centres, ws_counts, yerr=ws_err, fmt="k.")

    rs_predicted = rs_scale * pdfs.fractional_pdf(centres, *rs_params)
    ws_predicted = ws_scale * pdfs.fractional_pdf(centres, *ws_params)

    ax[0].plot(centres, rs_predicted)
    ax[1].plot(centres, ws_predicted)

    ax[0].plot(
        centres,
        rs_scale * rs_params[0] * pdfs.normalised_signal(centres, *rs_params[1:-2]),
        label="signal",
    )
    ax[1].plot(
        centres,
        ws_scale * ws_params[0] * pdfs.normalised_signal(centres, *ws_params[1:-2]),
        label="signal",
    )

    ax[0].plot(
        centres,
        rs_scale * (1 - rs_params[0]) * pdfs.normalised_bkg(centres, *rs_params[-2:]),
        label="bkg",
    )
    ax[1].plot(
        centres,
        ws_scale * (1 - ws_params[0]) * pdfs.normalised_bkg(centres, *ws_params[-2:]),
        label="bkg",
    )

    ax[0].set_title(r"RS $\Delta$ M")
    ax[1].set_title(r"WS $\Delta$ M")

    return fig, ax


def main():
    """
    Create plots

    """
    rs = _delta_m(pd.concat(get.data("2018", "cf", "magdown")))
    ws = _delta_m(pd.concat(get.data("2018", "dcs", "magdown")))

    fitter = simultaneous_fit(rs, ws, 5)
    fig, ax = _plot_fit(rs, ws, fitter.values)

    fig.tight_layout()

    # Get the number of background events from the background fraction * the number of WS events
    # ----
    # Note that this assumes we did the fit with "equivalent" amounts of CF and DCS data
    # if we have (e.g.) much more CF than DCS, then we will get the wrong answer out.
    # Perhaps it would just be best to do it with the whole sample, when actually optimising the BDT
    n_bkg = (1 - fitter.values[-1]) * len(ws)

    # Get the number of signal events from the signal fraction * number of RS events, scaled by the
    # approximately known amplitude ratio of 0.055
    n_sig = len(rs) * fitter.values[0] * 0.055 ** 2

    print(f"approx WS signal fraction {n_sig / (n_sig + n_bkg)}")

    plt.show()


if __name__ == "__main__":
    main()