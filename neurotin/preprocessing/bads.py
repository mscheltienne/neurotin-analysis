from typing import List

import mne
import numpy as np
import pyprep
from autoreject import Ransac
from mne.io import BaseRaw

from ..config.events import EVENTS, EVENTS_DURATION_MAPPING
from ..utils._docs import fill_doc
from .filters import apply_filter_eeg


def _prepapre_raw(raw: BaseRaw) -> BaseRaw:
    """Copy the raw instance and crops it based on the recording type.

    The cropping rules are:
        - Resting-State: Crop 2 minutes starting at the trigger.
        - Calibration: Crop from the first rest phase to the last stimuli.
        - Online: Crop from the first non-regulation to the last regulation.

    Set the montage as 'standard_1020'. The reference 'CPz' is not added.
    """
    apply_filter_eeg(raw, bandpass=(1.0, 40.0), notch=True)
    events = mne.find_events(raw, stim_channel="TRIGGER")
    unique_events = list(set(event[2] for event in events))

    # Calibration
    if EVENTS["audio"] in unique_events:
        sample_min, _, _ = events[
            np.where([ev[2] == EVENTS["rest"] for ev in events])
        ][0]
        tmin = sample_min / raw.info["sfreq"]
        sample_max, _, _ = events[
            np.where([ev[2] == EVENTS["audio"] for ev in events])
        ][-1]
        tmax = (
            sample_max / raw.info["sfreq"]
            + EVENTS_DURATION_MAPPING[EVENTS["audio"]]
        )
    # Resting-State
    elif EVENTS["resting-state"] in unique_events:
        sample_min, _, _ = events[
            np.where([ev[2] == EVENTS["resting-state"] for ev in events])
        ][0]
        tmin = sample_min / raw.info["sfreq"]
        tmax = tmin + EVENTS_DURATION_MAPPING[EVENTS["resting-state"]]
    # Online Run
    elif EVENTS["regulation"] in unique_events:
        sample_min, _, _ = events[
            np.where([ev[2] == EVENTS["non-regulation"] for ev in events])
        ][0]
        tmin = sample_min / raw.info["sfreq"]
        sample_max, _, _ = events[
            np.where([ev[2] == EVENTS["regulation"] for ev in events])
        ][-1]
        tmax = (
            sample_max / raw.info["sfreq"]
            + EVENTS_DURATION_MAPPING[EVENTS["regulation"]]
        )

    assert tmin < tmax
    raw.crop(tmin, tmax, include_tmax=True)
    raw.set_montage("standard_1020")

    return raw


@fill_doc
def RANSAC_bads_suggestion(
    raw: BaseRaw,
    prepare_raw: bool = True,
) -> List[str]:
    """Apply a RANSAC algorithm to detect bad channels using autoreject.

    Parameters
    ----------
    %(raw)s
    prepare_raw : bool
        If True, the provided raw is cropped and filtered.

    Returns
    -------
    bads : list
        List of bad channels.
    """
    raw = raw.copy()
    raw = _prepapre_raw(raw) if prepare_raw else raw
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True
    )
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)
    return ransac.bad_chs_


@fill_doc
def PREP_bads_suggestion(
    raw: BaseRaw,
    prepare_raw: bool = True,
) -> List[str]:
    """Apply the PREP pipeline to detect bad channels.

    The PREP pipeline uses:
        - SNR
        - Correlation
        - Deviation
        - HF Noise
        - NaN flat
        - RANSAC

    Parameters
    ----------
    %(raw)s
    prepare_raw : bool
        If True, the provided raw is cropped and filtered.

    Returns
    -------
    bads : list
        List of bad channels.
    """
    raw = raw.copy()
    raw = _prepapre_raw(raw) if prepare_raw else raw
    raw.pick_types(eeg=True)
    nc = pyprep.find_noisy_channels.NoisyChannels(raw)
    nc.find_all_bads()
    return nc.get_bads()
