import os

import numpy as np

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_subject

#%% create folders
os.makedirs(TFR_FOLDER / "multitaper", exist_ok=True)
os.makedirs(TFR_FOLDER / "multitaper" / "full", exist_ok=True)
os.makedirs(TFR_FOLDER / "multitaper" / "regular", exist_ok=True)
os.makedirs(TFR_FOLDER / "multitaper" / "transfer", exist_ok=True)

#%% multitaper settings
freqs = np.arange(1, 15, 1)
n_cycles = 2 * freqs
T = n_cycles / freqs
time_bandwidth = 4
fq_resolution = time_bandwidth / T  # +/- on both sides of the fq of interest

#%% subject TFR
for name, regular_only, transfer_only in [
    ("full", False, False),
    ("regular", True, False),
    ("transfer", False, True),
]:
    results = tfr_subject(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only=True,
        regular_only=regular_only,
        transfer_only=transfer_only,
        participants=PARTICIPANTS,
        method="multitaper",
        n_jobs=len(PARTICIPANTS),
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
    )

    directory = TFR_FOLDER / "multitaper" / name
    for subject, (tfr, itc) in results.items():
        tfr.save(directory / f"sub-{subject}-tfr.h5", overwrite=True)
        itc.save(directory / f"sub-{subject}-itc.h5", overwrite=True)
