"""
Read the background and signal points (and masses) into pandas DataFrames

"""
import pickle
from multiprocessing import Process
from typing import List
import pandas as pd
import numpy as np
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


def _create_df(tree, background: bool = False) -> pd.DataFrame:
    """
    Populate a pandas dataframe with the training variables, D0 mass and delta M

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
    return pd.DataFrame(
        np.column_stack((training_vars, d_mass, delta_m)),
        columns=[*read_data.training_var_names(), "D Mass", "Delta M"],
    )


def _create_dump(
    gen: np.random.Generator,
    sign: str,
    files: List[str],
    background: bool = False,
    train_fraction: float = 0.5,
) -> None:
    """
    Create pickle dump of the background points, D0 mass and delta M and test/train flag

    :param gen: random number generator for train/test split
    :param sign: "RS" or "WS" - tells us which tree to read from the file
    :param files: iterable of filepaths to read from
    :param background: bool flag telling us whether this data is signal or background - they get treated slightly
                       differently (apply straight cuts to signal; take only evts in upper mass sidebands for bkg)
    :param train_fraction: how much data to set aside for testing/training.

    """
    dfs = []
    for root_file in files:
        with uproot.open(root_file) as f:
            df = _create_df(f[definitions.tree_name(sign)], background)
            df["train"] = gen.random(len(df)) < train_fraction

            dfs.append(df)

    # Dump it
    path = definitions.df_dump_path(background)
    with open(path, "wb") as f:
        print(f"dumping {path}")
        pickle.dump(pd.concat(dfs), f)


def main(year: str, sign: str, background: bool) -> None:
    """
    Create pickle dump

    """
    files = (
        definitions.mc_files(year, "magdown")
        if not background
        else definitions.data_files(year, "magdown")
    )
    gen = np.random.default_rng(seed=0)
    _create_dump(gen, sign, files, background)


if __name__ == "__main__":
    procs = [
        Process(target=main, args=("2018", "RS", background))
        for background in (True, False)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
