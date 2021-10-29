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


def make_fixed_length_epochs(raw, duration=1., overlap=0.):
    """
    Create fixed length epochs for neurofeedback runs.

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
    events = mne.find_events(raw, stim_channel='TRIGGER')
    unique_events = set(ev[2] for ev in events)
    event_id = {EVENTS_MAPPING[value]: value for value in unique_events}

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
            stop += 7  # first rest phase extension
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
