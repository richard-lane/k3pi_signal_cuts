"""
Plot training variables used by the classifier

"""
import sys
import pathlib
import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import read_data


def _data_path() -> str:
    """
    Path to file containing real data

    """
    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    return str(data_dir / "data_example.root")


def _mc_path() -> str:
    """
    Path to file containing signal only MC

    """
    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "mc_2018_magdown"
    return str(data_dir / "00123274_00000001_1.charm_d02hhhh_dvntuple.root")


def _training_vars(data_path: str, tree_name: str) -> np.ndarray:
    """
    Array of training variables

    """
    with uproot.open(data_path) as f:
        tree = f[tree_name]
        points = read_data.points(tree)

    return points


def main():
    sign = "RS"

    # Calculate/read the training variables from the real data tree
    tree_name = (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )
    data_points = _training_vars(_data_path(), tree_name)

    # Calculate/read the training variables from the signal only MC tree
    mc_points = _training_vars(_mc_path(), tree_name)

    # Plot them
    fig, ax = plt.subplots(2, 7, figsize=(14, 4))
    for axis, data, mc, label in tqdm(
        zip(ax.ravel(), data_points.T, mc_points.T, read_data.training_var_names())
    ):
        bins = np.linspace(*np.quantile(data_points, [0.01, 0.99]), 100)
        axis.hist(data, bins=bins, histtype="step", density=True)
        axis.hist(mc, bins=bins, histtype="step", density=True)
        axis.set_title(label)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
