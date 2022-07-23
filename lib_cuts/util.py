"""
Utility functions

"""
import pickle
from . import definitions


def read_dataframe(background: bool = True):
    """
    Read a dataframe created by the `create_dumps.py` script

    :param background: bool flag telling us whether we want to read signal or background

    """
    try:
        with open(definitions.df_dump_path(background=background), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        print("=" * 79)
        print(
            f"Have you created the DataFrames by running the `create_dumps.py` script?"
        )
        print("=" * 79)
        raise e
