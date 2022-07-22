"""
Plot the signal peak used as input for the classifier

This comes from phase space MC with straight cuts applied

"""
import sys
import pathlib
import uproot
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))

from lib_cuts import cuts


def _masses_and_mask(sign: str):
    """
    Return D0 mass, delta M and keep mask

    """
    assert sign in {"RS", "WS"}

    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "mc_2018_magdown"
    tree_name = (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )
    root_path = str(data_dir / "00123274_00000001_1.charm_d02hhhh_dvntuple.root")

    with uproot.open(root_path) as f:
        tree = f[tree_name]

        d0_mass = cuts._d0_mass(tree)
        delta_m = tree["Dst_ReFit_M"].array()[:, 0] - d0_mass
        keep = cuts.keep(tree)

    return d0_mass, delta_m, keep


def main():
    """ Make a plot """
    sign = "RS"
    d0_mass, delta_m, keep = _masses_and_mask(sign)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    n_bins = 150
    bins = np.linspace(1775, 1975, n_bins)
    ax[0].hist(d0_mass[keep], bins=bins, label="signal")
    ax[0].hist(d0_mass[~keep], bins=bins, label="not signal", histtype="step")

    bins = np.linspace(130, 180, n_bins)
    ax[1].hist(delta_m[keep], bins=bins)
    ax[1].hist(delta_m[~keep], bins=bins, histtype="step")

    ax[0].set_title(r"$D^0$ Mass")
    ax[1].set_title(r"$\Delta M(D* - D^0)$")

    ax[0].legend()

    fig.suptitle(f"{sign} Phsp MC with/without straight cuts")

    plt.show()


if __name__ == "__main__":
    main()
