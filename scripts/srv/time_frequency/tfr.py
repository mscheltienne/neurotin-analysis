import os

import numpy as np

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_global, tfr_session, tfr_subject

#%% create folders
os.makedirs(TFR_FOLDER, exist_ok=True)
os.makedirs(TFR_FOLDER / "full", exist_ok=True)
os.makedirs(TFR_FOLDER / "regular", exist_ok=True)
os.makedirs(TFR_FOLDER / "transfer", exist_ok=True)
os.makedirs(TFR_FOLDER / "session-level", exist_ok=True)

#%% multitaper settings
freqs = np.arange(1, 15, 1)
n_cycles = 2 * freqs
T = n_cycles / freqs
time_bandwidth = 4
fq_resolution = time_bandwidth / T  # +/- on both sides of the fq of interest

#%% global TFR
for baseline in (None, (0, 8)):
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
            baseline=baseline,
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
        )
        baseline_str = str(baseline)
        if baseline is not None:
            baseline_str = baseline_str.replace(", ", "-")
        fname = TFR_FOLDER / name / f"global-{baseline_str}-tfr.h5"
        tfr.save(fname, overwrite=True)
        fig = tfr.plot(baseline=(0, 8), mode="mean", combine="mean")
        fig[0].savefig(fname.with_suffix(".svg"))

#%% subject TFR
for baseline in (None, (0, 8)):
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
            baseline=baseline,
            # n_jobs=len(PARTICIPANTS),
            n_jobs=4,
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
        )

        baseline_str = str(baseline)
        if baseline is not None:
            baseline_str = baseline_str.replace(", ", "-")

        for subject, tfr in results.items():
            fname = TFR_FOLDER / name / f"sub-{subject}-{baseline_str}-tfr.h5"
            tfr.save(fname, overwrite=True)
            fig = tfr.plot(baseline=(0, 8), mode="mean", combine="mean")
            fig[0].savefig(fname.with_suffix(".svg"))

#%% session TFR
for baseline in (None, (0, 8)):
    results = tfr_session(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only=True,
        participants=PARTICIPANTS,
        method="multitaper",
        baseline=baseline,
        # n_jobs=len(PARTICIPANTS),
        njobs=4,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
    )

    baseline_str = str(baseline)
    if baseline is not None:
        baseline_str = baseline_str.replace(", ", "-")

    for subject, session_tfrs in results.items():
        for session, tfr in session_tfrs.items():
            dir_ = TFR_FOLDER / "session-level"
            fname = dir_ / f"sub-{subject}-ses-{session}-{baseline_str}-tfr.h5"
            tfr.save(fname, overwrite=True)
            fig = tfr.plot(baseline=(0, 8), mode="mean", combine="mean")
            fig[0].savefig(fname.with_suffix(".svg"))
