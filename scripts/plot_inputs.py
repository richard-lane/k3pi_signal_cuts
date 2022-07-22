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


def _training_vars(sign: str) -> np.ndarray:
    """
    Array of training variables

    """
    assert sign in {"RS", "WS"}

    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    tree_name = (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )
    root_path = str(data_dir / "data_example.root")

    with uproot.open(root_path) as f:
        tree = f[tree_name]
        points = read_data.points(tree)

    return points


def main():
    # Calculate/read the training variables from the tree
    rs_points = _training_vars("RS")

    # Plot them
    fig, ax = plt.subplots(2, 7, figsize=(14, 4))
    for axis, points, label in tqdm(
        zip(ax.ravel(), rs_points.T, read_data.training_var_names())
    ):
        bins = np.linspace(*np.quantile(points, [0.01, 0.99]), 100)
        axis.hist(points, bins=bins)
        axis.set_title(label)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
