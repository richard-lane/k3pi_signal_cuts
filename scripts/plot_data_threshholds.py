"""
Plot delta M in data at various signal significances

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
sys.path.append(str(pathlib.Path(__file__).absolute().parents[2] / "k3pi-data"))

from lib_cuts.get import classifier as get_clf
from lib_data import get, training_vars


def _delta_m(data: pd.DataFrame) -> np.ndarray:
    """ D* - D0 Mass """
    return data["D* mass"] - data["D0 mass"]


def _plot(ax: plt.Axes, delta_m: np.ndarray, threshhold: float) -> None:
    """
    Plot delta M distribution on an axis

    """
    bins = np.linspace(139, 152, 120)
    counts, _ = np.histogram(delta_m, bins)
    centres = (bins[:-1] + bins[1:]) / 2
    ax.plot(centres, counts, label=f"{threshhold:.3f}")


def main():
    """
    Plot DCS delta M for various values of the cut threshhold

    For each, work out the signal significance

    """
    year, sign, magnetisation = "2018", "dcs", "magdown"
    ws_df = pd.concat(get.data(year, sign, magnetisation))

    clf = get_clf(year, sign, magnetisation)
    training_var_names = list(training_vars.training_var_names())
    sig_probs = clf.predict_proba(ws_df[training_var_names])[:, 1]

    # For various values of the threshhold, perform cuts
    # and plot the resultant delta M distribution
    x_range = 0.0, 1.0
    threshholds = np.sort(np.append(np.linspace(*x_range, 7), (0.019, 0.084)))

    delta_m = _delta_m(ws_df)
    fig, ax = plt.subplots()

    for threshhold in threshholds:
        _plot(ax, delta_m[sig_probs > threshhold], threshhold)

    fig.suptitle("WS data; BDT cuts at various probability threshholds")
    ax.legend(title="Threshhold")

    ax.text(146, 430, "optimal significance", color="g")
    ax.text(148, 680, "optimal FPR/TPR", color="orange", rotation=15)

    fig.savefig("data_threshholds.png")

    plt.show()


if __name__ == "__main__":
    main()
