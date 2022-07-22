"""
Plot training variables used by the classifier

"""
import sys
import pickle
import pathlib
import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data, definitions


def main():
    year, sign = "2018", "RS"
    # Read dataframes of stuff
    with open(definitions.df_dump_path(background=True), "rb") as f:
        bkg_df = pickle.load(f)

    with open(definitions.df_dump_path(background=False), "rb") as f:
        sig_df = pickle.load(f)

    # Plot
    fig, ax = plt.subplots(5, 3, figsize=(15, 9))

    kw = {"histtype": "step", "density": True}
    quantiles = [0.01, 0.99]  # Which quantiles to use for binning
    n_bins = 100
    for col, axis in zip(bkg_df, ax.ravel()):
        bkg_quantile = np.quantile(bkg_df[col], quantiles)
        sig_quantile = np.quantile(sig_df[col], quantiles)

        bins = np.linspace(
            min(bkg_quantile[0], sig_quantile[0]),
            max(bkg_quantile[1], sig_quantile[1]),
            n_bins,
        )

        axis.hist(sig_df[col], bins=bins, **kw, label="sig")
        axis.hist(bkg_df[col], bins=bins, **kw, label="bkg")
        axis.set_title(col)
        axis.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
