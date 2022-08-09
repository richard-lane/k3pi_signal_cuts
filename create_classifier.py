"""
Train a classifier to separate signal and background

You should have already downloaded the data files and created pickle dumps

"""
import os
import sys
import pickle
import pathlib
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from lib_cuts import definitions
from lib_cuts import util

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1] / "k3pi-data"))

from lib_data import get, training_vars


def _train_test_dfs(year, sign, magnetisation):
    """ Join signal + bkg dataframes but split by train/test """
    sig_df = get.mc(year, sign, magnetisation)
    bkg_df = pd.concat(get.uppermass(year, sign, magnetisation))

    combined_df = pd.concat((sig_df, bkg_df))

    # 1 for signal, 0 for background
    labels = np.concatenate((np.ones(len(sig_df)), np.zeros(len(bkg_df))))

    train_mask = combined_df["train"]
    return (
        combined_df[train_mask],
        combined_df[~train_mask],
        labels[train_mask],
        labels[~train_mask],
    )


def main(year: str, sign: str, magnetisation: str):
    """
    Create the classifier, print training scores

    """
    # If classifier already exists, tell us and exit
    clf_path = definitions.classifier_path(year, sign, magnetisation)
    if clf_path.is_file():
        print(f"{clf_path} exists")
        return

    classifier_dir = pathlib.Path(__file__).resolve().parents[0] / "classifiers"
    if not classifier_dir.is_dir():
        os.mkdir(str(classifier_dir))

    # Label 1 for signal; 0 for bkg
    train_df, test_df, train_label, test_label = _train_test_dfs(
        year, sign, magnetisation
    )

    # We want to train the classifier on a realistic proportion of signal + background
    # Get this from running `scripts/mass_fit.py`
    # using this number for now
    sig_frac = 0.5
    train_weights = util.weights(train_label, sig_frac)

    clf = RandomForestClassifier(n_estimators=200)
    # We only want to use some of our variables for training
    training_labels = list(training_vars.training_var_names())
    clf.fit(train_df[training_labels], train_label, train_weights)

    print(classification_report(train_label, clf.predict(train_df[training_labels])))

    # Resample for the training part of the classification report
    gen = np.random.default_rng()
    test_mask = util.resample_mask(gen, test_label, sig_frac)
    print(
        classification_report(
            test_label[test_mask], clf.predict(test_df[training_labels][test_mask])
        )
    )

    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation)
