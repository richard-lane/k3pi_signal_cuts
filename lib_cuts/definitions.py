"""
Useful definitions and things

"""
import os
import glob
import pathlib
from typing import List

CLASSIFIER_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "classifier.pkl")

BKG_DUMP = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "bkg_dump.pkl")

SIGNAL_DUMP = str(
    pathlib.Path(__file__).resolve().parents[1] / "data" / "signal_dump.pkl"
)

D0_MASS_MEV = 1864.84


def mc_files(year: str, magnetisation: str) -> List[str]:
    """
    Return a list of strings pointing to MC ROOT files that we have downloaded and put in the right directory

    For instructions on how to do this, see `data/README.md`

    """
    # Haven't got round to many of these
    assert year in {"2018"}
    assert magnetisation in {"magdown"}

    data_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / f"mc_{year}_{magnetisation}"
    )

    files = glob.glob(str(data_dir / "*"))

    if not files:
        raise FileNotFoundError(
            f"No files found in {data_dir}; have you downloaded them? See the readme in the `data/` dir."
        )

    return files


def data_files(year: str, magnetisation: str) -> List[str]:
    """
    Return a list of strings pointing to data ROOT files that we have downloaded and put in the right directory

    For instructions on how to do this, see `data/README.md`

    """
    # Haven't got round to many of these
    assert year in {"2018"}
    assert magnetisation in {"magdown"}

    data_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data"
        / f"data_{year}_{magnetisation}"
    )

    files = glob.glob(str(data_dir / "*"))

    if not files:
        raise FileNotFoundError(
            f"No files found in {data_dir}; have you downloaded them? See the readme in the `data/` dir."
        )

    return files


def tree_name(sign: str) -> str:
    """
    Name of tree in ROOT file

    """
    assert sign in {"RS", "WS"}

    return (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "RS"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )


def bkg_dump_exists() -> bool:
    """ Whether the data dump has been created """
    return os.path.exists(BKG_DUMP)


def signal_dump_exists() -> bool:
    """ Whether the data dump has been created """
    return os.path.exists(SIGNAL_DUMP)


def df_dump_path(background: bool) -> str:
    """
    Where the dataframe pickle dump lives

    background parameter is a bool flagging whether we want the location of sig or bkg

    """
    if background:
        return BKG_DUMP
    return SIGNAL_DUMP
