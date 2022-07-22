data dir
====

Running stuff
----
The classifier is trained on signal and background - the signal sample is phase space MC (see below).
Cuts are applied to make sure that this MC sample really does only contain signal - these can be seen in the
`lib_cuts/cuts.py` module, and an example of their use is in `scripts/plot_signal.py`.

The background sample is taken from the upper-mass sidebands of real data - see below.
There is much more real data than MC - see below for the locations of these files on lxplus and instructions
on downloading them if desired.

All the scripts, training code etc. in this repo assumes that the relevant data is downloaded and stored in the `data/`
dir - you might have to change some paths and things if you're storing your data differently or running on lxplus.

Monte Carlo
----
The locations of MC files on lxplus are found in the text files in this dir.
There aren't many of them - download them with `scp` or `rsync` (better) to run things locally.

The MC was generated according to phase space only (i.e. no amplitude model).

Real Data
----
To run the scripts in the `scripts/` dir, you will likely have to download the data from lxplus and put it into this
directory (`data/`)

The `k3pi_mass_fit` repo on GitHub contains scripts to download data ROOT files from lxplus - perhaps clone that repo,
download (some of) the files and copy the data to this directory (or just create a symlink).

Alternatively just go into the source file of the script you want to run and edit the appropriate path(s) to point to
whatever data you want to use.

