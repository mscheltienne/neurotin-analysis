import os

import numpy as np

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_global

#%% create folders
os.makedirs(TFR_FOLDER, exist_ok=True)
os.makedirs(TFR_FOLDER / "full", exist_ok=True)
os.makedirs(TFR_FOLDER / "regular", exist_ok=True)
os.makedirs(TFR_FOLDER / "transfer", exist_ok=True)

#%% multitaper settings
freqs = np.arange(1, 15, 1)
n_cycles = 2 * freqs
T = n_cycles / freqs
time_bandwidth = 4
fq_resolution = time_bandwidth / T  # +/- on both sides of the fq of interest

#%% global TFR
for name, regular_only, transfer_only in [
    ("full", False, False),
    ("regular", True, False),
    ("transfer", False, True),
]:
    tfr = tfr_global(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only=True,
        regular_only=regular_only,
        transfer_only=transfer_only,
        participants=PARTICIPANTS,
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
    )

    fname = TFR_FOLDER / name / "global-tfr.h5"
    tfr.save(fname, overwrite=True)
    fig = tfr.plot(baseline=(0, 8), mode="mean", combine="mean")
    fig[0].savefig(fname.with_suffix(".svg"))
