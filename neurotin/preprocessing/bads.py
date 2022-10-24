from typing import List

import mne
import pyprep
from autoreject import Ransac
from mne.io import BaseRaw

from ..utils._docs import fill_doc
from .events import find_crop_tmin_tmax
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
    tmin, tmax = find_crop_tmin_tmax(raw)
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
    nc = pyprep.find_noisy_channels.NoisyChannels(raw, do_detrend=False)
    nc.find_bad_by_SNR()
    nc.find_bad_by_correlation()
    nc.find_bad_by_hfnoise()
    nc.find_bad_by_nan_flat()
    nc.find_bad_by_ransac()
    return nc.get_bads()
