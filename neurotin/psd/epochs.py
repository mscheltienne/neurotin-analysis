from itertools import chain
from typing import Dict, Tuple, Union

import mne
import numpy as np
from autoreject import AutoReject, get_rejection_threshold
from mne import BaseEpochs
from mne.io import BaseRaw, RawArray
from numpy.typing import NDArray

from ..config.events import (
    EVENTS,
    EVENTS_DURATION_MAPPING,
    EVENTS_MAPPING,
    FIRST_REST_PHASE_EXT,
)
from ..utils._checks import _check_type, _check_value
from ..utils._docs import fill_doc


@fill_doc
def make_epochs(raw: BaseRaw) -> Dict[str, BaseEpochs]:
    """Create epochs for regulation and non-regulation events.

    Regulation epochs last 16 seconds. Non-regulation epochs last 8 seconds.
    The first non-regulation epoch is cropped around its last 8 seconds.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    epochs : dict of Epochs
        Epochs for 'regulation' and for 'non-regulation'.
    """
    _check_type(raw, (BaseRaw,), item_name="raw")

    # load events
    events, event_id = _load_events(raw)

    # change event sample of the first non-regulation phase
    events[0, 0] += FIRST_REST_PHASE_EXT * raw.info["sfreq"]

    # create epochs
    epochs = dict()
    for ev, idx in event_id.items():
        epochs[ev] = mne.Epochs(
            raw,
            events=events,
            event_id={ev: idx},
            tmin=0.0,
            tmax=EVENTS_DURATION_MAPPING[idx],
            baseline=None,
            picks="eeg",
            preload=True,
            reject=None,
            flat=None,
        )

    return epochs


@fill_doc
def make_fixed_length_epochs(
    raw: BaseRaw, duration: float = 4.0, overlap: float = 3.0
) -> BaseEpochs:
    """Create fixed length epochs for neurofeedback runs.

    Aggregate epochs  from the same phase together.
        non-regulation-0:   60
        regulation-0:       50
        non-regulation-1:   61
        regulation-1:       51
        ...                 ..
        non-regulation-9:   69
        regulation-9:       59

    Parameters
    ----------
    %(raw)s
    %(psd_duration)s
    %(psd_overlap)s

    Returns
    -------
    epochs : Epochs
    """
    _check_type(raw, (BaseRaw,), item_name="raw")
    _check_type(duration, ("numeric",), item_name="duration")
    _check_type(overlap, ("numeric",), item_name="overlap")

    # load events
    events, event_id = _load_events(raw)

    # add new stim channel
    raw = raw.copy()
    info = mne.create_info(["STI"], sfreq=raw.info["sfreq"], ch_types="stim")
    stim = RawArray(np.zeros(shape=(1, len(raw.times))), info)
    raw.add_channels([stim], force_update_info=True)

    # add fixed length events to the new stim channel
    for k, event in enumerate(events):
        start = event[0] / raw.info["sfreq"]
        stop = start + EVENTS_DURATION_MAPPING[event[2]]
        if k == 0:
            stop += FIRST_REST_PHASE_EXT  # first rest phase extension
        epoch_events = mne.make_fixed_length_events(
            raw,
            id=int(event[2] * 10 + k // 2),
            start=start,
            stop=stop,
            duration=duration,
            first_samp=False,
            overlap=overlap,
        )
        raw.add_events(epoch_events, stim_channel="STI", replace=False)

    # create epochs from the new stim channel
    event_ids = [
        {
            key + f"-{k // 2}": event_id[key] * 10 + k // 2
            for k in range(events.shape[0])
        }
        for key in event_id
    ]
    event_ids = dict(chain(*map(dict.items, event_ids)))
    events = mne.find_events(raw, stim_channel="STI")
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_ids,
        tmin=0.0,
        tmax=duration - 1 / raw.info["sfreq"],
        baseline=None,
        picks="eeg",
        preload=True,
        reject=None,
        flat=None,
    )

    return epochs


def _load_events(raw: BaseRaw) -> Tuple[NDArray[int], Dict[str, int]]:
    """Load events from raw instance and check if it is an online run."""
    events = mne.find_events(raw, stim_channel="TRIGGER")
    assert events.shape == (20, 3)
    unique_events = set(ev[2] for ev in events)
    assert unique_events == set(
        (EVENTS["regulation"], EVENTS["non-regulation"])
    )
    event_id = {EVENTS_MAPPING[value]: value for value in unique_events}
    return events, event_id


@fill_doc
def reject_epochs(
    epochs: BaseEpochs, reject: Union[Dict[str, float], str] = "auto"
) -> Tuple[BaseEpochs, Dict[str, float]]:
    """Reject bad epochs with a global rejection threshold.

    Parameters
    ----------
    epochs : Epochs
        Raw epochs, before peak-to-peak rejection.
    %(psd_reject)s

    Returns
    -------
    epochs : Epochs
        Good epochs.
    reject : dict
        Rejection dictionary used to drop epochs.
    """
    _check_type(epochs, (BaseEpochs,), item_name="epochs")
    _check_type(reject, (dict, str), item_name="reject")
    if isinstance(reject, str):
        _check_value(reject, ("auto",), item_name="reject")
    elif isinstance(reject, dict):
        assert len(reject) == 1 and "eeg" in reject

    if reject == "auto":
        reject = get_rejection_threshold(epochs, decim=1)
    epochs.drop_bad(reject=reject)
    return epochs, reject


def repair_epochs(
    epochs: BaseEpochs, thresh_method: str = "random_search"
) -> BaseEpochs:
    """Repair bad epochs using autoreject.

    Parameters
    ----------
    epochs : Epochs
        Epochs used to fit the rejection model and to repair with the model.
    thresh_method : str
        Either 'random_search' or 'bayesian_optimization'.

    Returns
    -------
    epochs : Epochs
        Epochs repaired by autoreject.
    """
    _check_type(epochs, (BaseEpochs,), item_name="epochs")
    _check_value(
        thresh_method,
        ("random_search", "bayesian_optimization"),
        item_name="thresh_method",
    )

    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)

    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    ar = AutoReject(
        n_interpolates,
        consensus_percs,
        picks=picks,
        thresh_method=thresh_method,
    )

    ar.fit(epochs)
    return ar.transform(epochs)
