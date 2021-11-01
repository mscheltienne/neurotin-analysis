from pathlib import Path

import mne
import numpy as np
from autoreject import AutoReject, get_rejection_threshold


EVENTS = {
    "rest": 1,
    "blink": 2,
    "resting-state": 3,
    "audio": 4,
    "regulation": 5,
    "non-regulation": 6,
}
EVENTS_MAPPING = {
    1: "rest",
    2: "blink",
    3: "resting-state",
    4: "audio",
    5: "regulation",
    6: "non-regulation",
}
EVENTS_DURATION_MAPPING = {
    1: 1,
    2: 60,
    3: 120,
    4: 0.8,
    5: 16,
    6: 8,
}
FIRST_REST_PHASE_EXT = 7  # extension of the first rest phase in seconds.


def make_fixed_length_epochs(raw, duration=1., overlap=0.):
    """
    Create fixed length epochs for neurofeedback runs and aggregate epochs
    together in 2 categories: regulation and non-regulation.

    Parameters
    ----------
    raw : Raw
        Preprocessed raw instance.
    duration : float
        Duration of an epoch in seconds.
    overlap : float
        Duration of epoch overlap in seconds.

    Returns
    -------
    epochs : Epochs
    """
    # load events
    events, event_id = _load_events(raw)

    # add new stim channel
    raw = raw.copy()
    info = mne.create_info(['STI'], sfreq=raw.info['sfreq'], ch_types='stim')
    stim = mne.io.RawArray(np.zeros(shape=(1, len(raw.times))), info)
    raw.add_channels([stim], force_update_info=True)

    # add fixed length events to the new stim channel
    for k, event in enumerate(events):
        start = event[0] / raw.info['sfreq']
        stop = start + EVENTS_DURATION_MAPPING[event[2]]
        if k == 0:
            stop += FIRST_REST_PHASE_EXT  # first rest phase extension
        epoch_events = mne.make_fixed_length_events(
            raw, id=int(event[2]), start=start, stop=stop,
            duration=duration, first_samp=False, overlap=overlap)
        raw.add_events(epoch_events, stim_channel='STI', replace=False)

    # create epochs from the new stim channel
    events = mne.find_events(raw, stim_channel='STI')
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=0.,
                        tmax=duration, baseline=None, picks='eeg',
                        preload=True, reject=None, flat=None)

    return epochs


def make_epochs(raw):
    """
    Create epochs for regulation and non-regulation events.
    Regulation epochs last 16 seconds.
    Non-regulation epochs last 8 seconds. The first non-regulation epoch is
    cropped on its last 8 seconds.

    Parameters
    ----------
    raw : Raw
        Preprocessed raw instance.

    Returns
    -------
    epochs : dict of Epochs
        Epochs for 'regulation' and for 'non-regulation'.
    """
    # load events
    events, event_id = _load_events(raw)

    # change event sample of the first non-regulation phase
    events[0, 0] += FIRST_REST_PHASE_EXT * raw.info['sfreq']

    # create epochs
    epochs = dict()
    for ev, idx in event_id.items():
        epochs[ev] = mne.Epochs(raw, events=events, event_id={ev: idx},
                                tmin=0., tmax=EVENTS_DURATION_MAPPING[idx],
                                baseline=None, picks='eeg', preload=True,
                                reject=None, flat=None)

    return epochs


def _load_events(raw):
    """
    Load events from raw instance and check if it is an online run.
    """
    events = mne.find_events(raw, stim_channel='TRIGGER')
    assert events.shape == (20, 3)
    unique_events = set(ev[2] for ev in events)
    assert unique_events == set((EVENTS['regulation'],
                                 EVENTS['non-regulation']))
    event_id = {EVENTS_MAPPING[value]: value for value in unique_events}

    return events, event_id


def reject_epochs(epochs, reject=None):
    """
    Reject bad epochs with a global rejection threshold.

    Parameters
    ----------
    epochs : Epochs
        Raw epochs, before peak-to-peak rejection.
    reject : dict | None
        MNE-compatible rejection dictionary or None to compute it with
        autoreject.

    Returns
    -------
    epochs : Epochs
        Good epochs.
    reject : dict
        Rejection dictionary used to drop epochs.
    """
    if reject is None:
        reject = get_rejection_threshold(epochs, decim=1)
    epochs.drop_bad(reject=reject)
    return epochs, reject


def repair_epochs(epochs, thresh_method='random_search'):
    """
    Repair bad epochs using autoreject.

    Parameters
    ----------
    epochs : Epochs
        Epochs used to fit the rejection model and to repair with the model.
    thresh_method : str
        Either 'random_search' or 'bayesian_optimization'.

    Returns
    -------
    epochs : Epochs
        Epochs repaired by the model.
    """
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)

    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                    thresh_method=thresh_method)

    ar.fit(epochs)
    return ar.transform(epochs)


def list_raw_fif(directory, exclude=[]):
    """
    List all raw fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory.
    exclude : list | tuple
        List of files to exclude.

    Returns
    -------
    fifs : list
        Found raw fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_raw_fif(elt))
        elif elt.name.endswith("-raw.fif") and elt not in exclude:
            fifs.append(elt)
    return fifs
