import mne
from autoreject import Ransac

from annotations import EVENTS_MAPPING


EVENTS_EPOCH_TIMINGS = {
    'rest': (0, 1),
    'blink': (0, 60),
    'resting-state': (0, 120),
    'audio': (-0.2, 0.8),
    'regulation': (0, 16),
    'non-regulation': (0, 8)
    }


def RANSAC_bads_suggestion(raw):
    """
    Create epochs around each found events and apply a RANSAC algorithm to
    detect bad channels.

    Parameters
    ----------
    raw : Raw
        Raw instance modified.

    Returns
    -------
    bads : dict
        event: list of bad channels
    unique_bads : list
        list of bad channels across all events
    """
    events = mne.find_events(raw, stim_channel='TRIGGER')
    unique_events = list(set(event[2] for event in events))

    bads = dict()
    for event in unique_events:
        tmin, tmax = EVENTS_EPOCH_TIMINGS[EVENTS_MAPPING[event]]
        baseline = (None, 0) if tmin < 0 else None
        epochs = mne.Epochs(
            raw, events, event_id=event, picks='eeg', tmin=tmin, tmax=tmax,
            reject=None, verbose=False, detrend=None, proj=True,
            baseline=baseline, preload=True)

        ransac = Ransac(verbose=False, picks='eeg', n_jobs=1)
        ransac.fit(epochs)
        bads[event] = ransac.bad_chs_

    return bads, list(set(bads.values()))
