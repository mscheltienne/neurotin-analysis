import os

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, DATA_PP_FOLDER, TFR_FOLDER
from neurotin.time_frequency import tfr_subject

#%% create folders
os.makedirs(TFR_FOLDER, exist_ok=True)
os.makedirs(TFR_FOLDER / "subject", exist_ok=True)

#%% full
valid_only = True
regular_only = False
transfer_only = False

tfrs = tfr_subject(
    DATA_FOLDER,
    DATA_PP_FOLDER,
    valid_only,
    regular_only,
    transfer_only,
    PARTICIPANTS,
    n_jobs=len(PARTICIPANTS),
)
for participant, tfr in tfrs.items():
    fname = TFR_FOLDER / "subject" / f"{str(participant).zfill(3)}-full.h5"
    tfr.save(fname, overwrite=True)
    fig = tfr.plot(combine="mean", show=False)[0]
    fig.savefig(fname.with_suffix(".svg"))

#%% regular
valid_only = True
regular_only = True
transfer_only = False

tfrs = tfr_subject(
    DATA_FOLDER,
    DATA_PP_FOLDER,
    valid_only,
    regular_only,
    transfer_only,
    PARTICIPANTS,
    n_jobs=len(PARTICIPANTS),
)
for participant, tfr in tfrs.items():
    fname = TFR_FOLDER / "subject" / f"{str(participant).zfill(3)}-regular.h5"
    tfr.save(fname, overwrite=True)
    fig = tfr.plot(combine="mean", show=False)[0]
    fig.savefig(fname.with_suffix(".svg"))

#%% transfer
valid_only = True
regular_only = False
transfer_only = True

tfrs = tfr_subject(
    DATA_FOLDER,
    DATA_PP_FOLDER,
    valid_only,
    regular_only,
    transfer_only,
    PARTICIPANTS,
    n_jobs=len(PARTICIPANTS),
)
for participant, tfr in tfrs.items():
    fname = TFR_FOLDER / "subject" / f"{str(participant).zfill(3)}-transfer.h5"
    tfr.save(fname, overwrite=True)
    fig = tfr.plot(combine="mean", show=False)[0]
    fig.savefig(fname.with_suffix(".svg"))
