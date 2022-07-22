"""
Read the background and signal points (and masses) into pandas DataFrames

"""
import os
import pickle
import pathlib
import pandas as pd
import numpy as np
from multiprocessing import Process
import uproot

from lib_cuts import read_data, definitions, cuts


def _bkg_keep(d_mass: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """
    Mask of events to keep after background mass cuts

    """
    # Keep points far enough away from the nominal mass
    d_mass_width = 24
    d_mass_range = (
        definitions.D0_MASS_MEV - d_mass_width,
        definitions.D0_MASS_MEV + d_mass_width,
    )
    d_mass_mask = (d_mass < d_mass_range[0]) | (d_mass_range[1] < d_mass)

    # AND within the delta M upper mass sideband
    delta_m_mask = (152 < delta_m) & (delta_m < 157)

    return d_mass_mask & delta_m_mask


def create_dump(tree, background: bool = False) -> None:
    """
    Create pickle dump of the background points, D0 mass and delta M

    """
    # Find training variables
    training_vars = read_data.points(tree)

    # Find D and D* masses
    d_mass = cuts.d0_mass(tree)
    delta_m = cuts.dst_mass(tree) - d_mass

    # If background, only use data from the upper mass sidebands
    if background:
        keep = _bkg_keep(d_mass, delta_m)

    # If signal, perform straight cuts
    else:
        keep = cuts.keep(tree)

    training_vars = training_vars[keep]
    d_mass = d_mass[keep]
    delta_m = delta_m[keep]

    # Populate dataframe
    df = pd.DataFrame(
        np.column_stack((training_vars, d_mass, delta_m)),
        columns=[*read_data.training_var_names(), "D Mass", "Delta M"],
    )

    # Dump it
    path = definitions.df_dump_path(background)
    with open(path, "wb") as f:
        print(f"dumping {path}")
        pickle.dump(df, f)


def main(year: str, sign: str, background: bool) -> None:
    """
    Create pickle dump

    """
    files = (
        definitions.mc_files(year, "magdown")
        if not background
        else definitions.data_files(year, "magdown")
    )
    with uproot.open(files[0]) as root_file:
        create_dump(root_file[definitions.tree_name(sign)], background)


if __name__ == "__main__":
    year, sign = "2018", "RS"
    main(year, sign, background=False)
    main(year, sign, background=True)
