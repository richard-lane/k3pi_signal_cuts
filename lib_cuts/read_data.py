"""
Read the right branches, do the right stuff to the data
to get it into the right format for the classifier

Some branches we can just read directly; others require some
calculation (e.g. the sum of daughter pT) before they can be used

"""
import numpy as np
import uproot  # type: ignore


def _refit_chi2(tree) -> np.ndarray:
    """
    The chi2 for the ReFit fit

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of ReFit values.
              Takes the first (best-fit) value from the jagged array

    """
    # Jagged array; take first (best-fit) value
    return tree["Dst_ReFit_chi2"].array()[:, 0]


def _endvertex_chi2(tree) -> np.ndarray:
    """
    D0 end vertex chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of chi2 values.

    """
    return tree["D0_ENDVERTEX_CHI2"].array()


def _orivx_chi2(tree) -> np.ndarray:
    """
    D0 origin vertex chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of chi2 values.

    """
    return tree["D0_ORIVX_CHI2"].array()


def _slow_pi_prob_nn_pi(tree) -> np.ndarray:
    """
    Neural network pion probability for soft pion.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of pion probabilities

    """
    return tree["Dst_pi_ProbNNpi"].array()


def _d0_pt(tree) -> np.ndarray:
    """
    ReFit D0 pT, calculated from daughter refit momenta.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    raise NotImplementedError


def _slow_pi_pt(tree) -> np.ndarray:
    """
    ReFit slow pi, calculated from slow pi momentum.

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    raise NotImplementedError


def _slow_pi_ipchi2(tree) -> np.ndarray:
    """
    slow pi IP chi2

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of D0 refit pT

    """
    return tree["Dst_pi_IPCHI2_OWNPV"].array()


def _pions_max_pt(tree) -> np.ndarray:
    """
    Max pT of daughter pions

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pions max pT

    """
    raise NotImplementedError


def _pions_min_pt(tree) -> np.ndarray:
    """
    Min pT of daughter pions

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pions min pT

    """
    raise NotImplementedError


def _pions_sum_pt(tree) -> np.ndarray:
    """
    Scalar sum of daughter pions pT

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter pion pT sum

    """
    raise NotImplementedError


def _daughters_max_pt(tree) -> np.ndarray:
    """
    Max pT of daughter particles

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles max pT

    """
    raise NotImplementedError


def _daughters_min_pt(tree) -> np.ndarray:
    """
    Min pT of daughter particles

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles min pT

    """
    raise NotImplementedError


def _daughters_sum_pt(tree) -> np.ndarray:
    """
    Scalar sum of daughter particles pT

    :param tree: real data uproot TTree to read from
                 (type annotation didnt work so i skipped it)
    :returns: 1d numpy array of daughter particles pT sum

    """
    raise NotImplementedError
