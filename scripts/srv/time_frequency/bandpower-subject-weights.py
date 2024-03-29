from neurotin.config import PARTICIPANTS
from neurotin.config.srv import (
    BP_FOLDER,
    DATA_FOLDER,
    DATA_PP_FOLDER,
    MODEL_FOLDER,
)
from neurotin.time_frequency import compute_bandpower_onrun

#%% constants
duration = 2.0
overlap = 1.9
frequencies = dict(alpha=(8.0, 13.0), delta=(1.0, 4.0))
n_jobs = 20


#%% full
valid_only = True
regular_only = False
transfer_only = False

for band, (fmin, fmax) in frequencies.items():
    df_fname = BP_FOLDER / f"{band}-onrun-full-subject-weights.pcl"
    df_abs, df_rel = compute_bandpower_onrun(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only,
        regular_only,
        transfer_only,
        PARTICIPANTS,
        fmin,
        fmax,
        duration,
        overlap,
        MODEL_FOLDER,
        "subject",
        n_jobs=n_jobs,
    )
    df_abs.to_pickle(
        df_fname.with_stem(df_fname.stem + "-abs"), compression=None
    )
    df_rel.to_pickle(
        df_fname.with_stem(df_fname.stem + "-rel"), compression=None
    )


#%% regular
valid_only = True
regular_only = True
transfer_only = False

for band, (fmin, fmax) in frequencies.items():
    df_fname = BP_FOLDER / f"{band}-onrun-regular-subject-weights.pcl"
    df_abs, df_rel = compute_bandpower_onrun(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only,
        regular_only,
        transfer_only,
        PARTICIPANTS,
        fmin,
        fmax,
        duration,
        overlap,
        MODEL_FOLDER,
        "subject",
        n_jobs=n_jobs,
    )
    df_abs.to_pickle(
        df_fname.with_stem(df_fname.stem + "-abs"), compression=None
    )
    df_rel.to_pickle(
        df_fname.with_stem(df_fname.stem + "-rel"), compression=None
    )

#%% transfer
valid_only = True
regular_only = False
transfer_only = True

for band, (fmin, fmax) in frequencies.items():
    df_fname = BP_FOLDER / f"{band}-onrun-transfer-subject-weights.pcl"
    df_abs, df_rel = compute_bandpower_onrun(
        DATA_FOLDER,
        DATA_PP_FOLDER,
        valid_only,
        regular_only,
        transfer_only,
        PARTICIPANTS,
        fmin,
        fmax,
        duration,
        overlap,
        MODEL_FOLDER,
        "subject",
        n_jobs=n_jobs,
    )
    df_abs.to_pickle(
        df_fname.with_stem(df_fname.stem + "-abs"), compression=None
    )
    df_rel.to_pickle(
        df_fname.with_stem(df_fname.stem + "-rel"), compression=None
    )
