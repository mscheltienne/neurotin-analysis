import os

import numpy as np

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_session_groupby

#%% create folders
os.makedirs(TFR_FOLDER, exist_ok=True)
os.makedirs(
    TFR_FOLDER / "multitaper" / "session-level-groupby-3", exist_ok=True
)
os.makedirs(
    TFR_FOLDER / "multitaper" / "session-level-groupby-5", exist_ok=True
)

#%% multitaper settings
freqs = np.arange(1, 15, 1)
n_cycles = 2 * freqs
T = n_cycles / freqs
time_bandwidth = 4
fq_resolution = time_bandwidth / T  # +/- on both sides of the fq of interest

#%% session TFR group-by 3
results = tfr_session_groupby(
    DATA_FOLDER,
    DATA_PP_FOLDER,
    valid_only=True,
    participants=PARTICIPANTS,
    method="multitaper",
    n_jobs=len(PARTICIPANTS),
    groupby=3,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
)

dir_ = TFR_FOLDER / "multitaper" / "session-level-groupby-3"
for subject, session_tfrs in results.items():
    for session, (tfr, itc) in session_tfrs.items():
        tfr.save(dir_ / f"sub-{subject}-{session}-tfr.h5", overwrite=True)
        itc.save(dir_ / f"sub-{subject}-{session}-itc.h5", overwrite=True)

#%% session TFR group-by 5
results = tfr_session_groupby(
    DATA_FOLDER,
    DATA_PP_FOLDER,
    valid_only=True,
    participants=PARTICIPANTS,
    method="multitaper",
    n_jobs=len(PARTICIPANTS),
    groupby=5,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
)

dir_ = TFR_FOLDER / "multitaper" / "session-level-groupby-5"
for subject, session_tfrs in results.items():
    for session, (tfr, itc) in session_tfrs.items():
        tfr.save(dir_ / f"sub-{subject}-{session}-tfr.h5", overwrite=True)
        itc.save(dir_ / f"sub-{subject}-{session}-itc.h5", overwrite=True)
