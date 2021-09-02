import mne
import numpy as np
from autoreject import Ransac

from events import EVENTS


def RANSAC_bads_suggestion(raw):
    """
    Create epochs around each found paradigm and apply a RANSAC algorithm to
    detect bad channels.

    Parameters
    ----------
    raw : Raw
        Raw instance.

    Returns
    -------
    bads : list
        Bad channels.
    """
    raw = raw.copy()

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

    raw.crop(tmin, tmax, include_tmax=True)
    raw.set_montage('standard_1020')
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True)
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)

    return ransac.bad_chs_
