import mne
import numpy as np
from autoreject import Ransac
from pyprep.find_noisy_channels import NoisyChannels

from events import EVENTS
from filters import apply_filter_eeg


def _prepapre_raw(raw):
    """
    Copy the raw instance and crops it based on the recording type:
        - Resting-State: Crop 2 minutes starting at the trigger.
        - Calibration: Crop from the first rest phase to the last stimuli.
        - Online: Crop from the first non-regulation to the last regulation.
    Set the montage as 'standard_1020'. The reference 'CPz' is not added.

    Parameters
    ----------
    raw : Raw
        Raw instance.

    Returns
    -------
    raw : Raw
        Copied and modified raw instance.
    """
    raw = raw.copy()
    apply_filter_eeg(raw, bandpass=(1., 40.), notch=True, car=False)
    events = mne.find_events(raw, stim_channel='TRIGGER')
    unique_events = list(set(event[2] for event in events))

    # Calibration
    if EVENTS['audio'] in unique_events:
        sample_min, _, _ = \
            events[np.where([ev[2]==EVENTS['rest'] for ev in events])][0]
        tmin = sample_min / raw.info['sfreq']
        sample_max, _, _ = \
            events[np.where([ev[2]==EVENTS['audio'] for ev in events])][-1]
        tmax = sample_max / raw.info['sfreq'] + 0.8
    # Resting-State
    elif EVENTS['resting-state'] in unique_events:
        sample_min, _, _ = events[np.where(
            [ev[2]==EVENTS['resting-state'] for ev in events])][0]
        tmin = sample_min / raw.info['sfreq']
        tmax = tmin + 120
    # Online Run
    elif EVENTS['regulation'] in unique_events:
        sample_min, _, _ = events[np.where(
            [ev[2]==EVENTS['non-regulation'] for ev in events])][0]
        tmin = sample_min / raw.info['sfreq']
        sample_max, _, _ = events[np.where(
            [ev[2]==EVENTS['regulation'] for ev in events])][-1]
        tmax = sample_max / raw.info['sfreq'] + 16

    try:
        assert tmin < tmax
    except Exception:
        raise AssertionError

    raw.crop(tmin, tmax, include_tmax=True)
    raw.set_montage('standard_1020')

    return raw


def RANSAC_bads_suggestion(raw):
    """
    Create fix length-epochs and apply a RANSAC algorithm to detect bad
    channels using autoreject.

    Parameters
    ----------
    raw : Raw
        Raw instance.

    Returns
    -------
    bads : list
        Bad channels.
    """
    raw = _prepapre_raw(raw)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True)
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)

    return ransac.bad_chs_


def PREP_bads_suggestion(raw):
    """
    Apply the PREP pipeline to detect bad channels:
        - SNR
        - Correlation
        - Deviation
        - HF Noise
        - NaN flat
        - RANSAC

    Parameters
    ----------
    raw : Raw
        Raw instance.

    Returns
    -------
    bads : list
        Bad channels.
    """
    raw = _prepapre_raw(raw)
    raw.pick_types(eeg=True)
    nc = NoisyChannels(raw)
    nc.find_all_bads()
    return nc.get_bads()
