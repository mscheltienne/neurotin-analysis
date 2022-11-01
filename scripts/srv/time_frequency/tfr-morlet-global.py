import os

import numpy as np

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_global

#%% create folders
os.makedirs(TFR_FOLDER / "morlet", exist_ok=True)
os.makedirs(TFR_FOLDER / "morlet" / "full", exist_ok=True)
os.makedirs(TFR_FOLDER / "morlet" / "regular", exist_ok=True)
os.makedirs(TFR_FOLDER / "morlet" / "transfer", exist_ok=True)

#%% multitaper settings
freqs = np.arange(1, 15, 1)
n_cycles = 2 * freqs
T = n_cycles / freqs

#%% global TFR
for name, regular_only, transfer_only in [
    ("full", False, False),
    ("regular", True, False),
    ("transfer", False, True),
]:
    tfr, itc = tfr_global(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only=True,
        regular_only=regular_only,
        transfer_only=transfer_only,
        participants=PARTICIPANTS,
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
    )

    directory = TFR_FOLDER / "morlet" / name
    tfr.save(directory / "global-tfr.h5", overwrite=True)
    itc.save(directory / "global-itc.h5", overwrite=True)
