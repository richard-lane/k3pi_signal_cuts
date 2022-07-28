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

    # Find also decay times
    times = read_data.decay_times(tree)

    # If background, only use data from the upper mass sidebands
    if background:
        keep = _bkg_keep(d_mass, delta_m)

    # If signal, perform straight cuts
    else:
        keep = cuts.keep(tree)

    training_vars = training_vars[keep]
    d_mass = d_mass[keep]
    delta_m = delta_m[keep]
    times = times[keep]

    # Populate dataframe
    return pd.DataFrame(
        np.column_stack((training_vars, d_mass, delta_m, times)),
        columns=[*read_data.training_var_names(), "D Mass", "Delta M", "time"],
    )


def _create_dump(
    gen: np.random.Generator,
    files: List[str],
    sign: str = None,
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
    # If a sign was provided we're creating a dump of background
    background = True if sign else False

    # Use the RS tree if we're reading signal MC (?) TODO work out if this is right
    # Otherwise use the tree that we asked for
    tree_name = definitions.tree_name(sign) if sign else definitions.tree_name("RS")

    dfs = []
    for root_file in files:
        with uproot.open(root_file) as f:
            df = _create_df(f[tree_name], background)
            df["train"] = gen.random(len(df)) < train_fraction

            dfs.append(df)

    # Dump it
    path = definitions.df_dump_path(background)
    with open(path, "wb") as f:
        print(f"dumping {path}")
        pickle.dump(pd.concat(dfs), f)


def main(year: str, bkg_sign: str = None) -> None:
    """
    Create pickle dump

    :param year: data-taking year
    :param bkg_sign: sign ("RS" or "WS") used for estimating the background.
                     Note that the the signal component used for the classifier cuts is taken from phase space MC,
                     so in this case we don't need to specify a sign.
                     Pass None here to create a dump for the signal.

    """
    # TODO change this perhaps
    gen = np.random.default_rng(seed=0)

    # Signal
    if not bkg_sign:
        _create_dump(gen, definitions.mc_files(year, "magdown", "phsp"))

    # Background
    else:
        _create_dump(gen, definitions.data_files(year, "magdown"), bkg_sign)


if __name__ == "__main__":
    procs = [
        Process(target=main, args=("2018",)),  # Signal
        Process(target=main, args=("2018", "RS")),  # Background
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
