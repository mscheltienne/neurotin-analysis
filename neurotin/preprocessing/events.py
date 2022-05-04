from collections import Counter
from typing import List, Optional, Tuple, Union

import mne
import numpy as np
from mne import Annotations
from mne.epochs import BaseEpochs
from mne.io import BaseRaw
from numpy.typing import NDArray

from ..config.events import EVENTS, EVENTS_DURATION_MAPPING, EVENTS_MAPPING
from ..utils._checks import _check_type, _check_value
from ..utils._docs import fill_doc


# ----------------------------------------------------------------------------
def find_event_channel(
    inst: Optional[Union[BaseRaw, BaseEpochs, NDArray[float]]] = None,
    ch_names: Optional[List[str]] = None,
) -> Optional[int, List[int]]:
    """
    Find the event channel using heuristics.

    .. warning::

        Not 100% guaranteed to find it.
        If ``inst`` is ``None``, ``ch_names`` must be given.
        If ``inst`` is an MNE instance, ``ch_names`` is ignored if some
        channels types are ``'stim'``.

    Parameters
    ----------
    inst : Raw | Epochs | `~numpy.array` | None
        Data instance. If a `~numpy.array` is provided, the shape must be
        ``(n_channels, n_samples)``.
    ch_names : list | None
        Channels name list.

    Returns
    -------
    event_channel : int | list | None
        Event channel index, list of event channel indexes or ``None`` if not
        found.
    """
    _check_type(
        inst, (None, np.ndarray, BaseRaw, BaseEpochs), item_name="inst"
    )
    _check_type(ch_names, (None, list, tuple), item_name="ch_names")

    # numpy array + ch_names
    if isinstance(inst, np.ndarray) and ch_names is not None:
        tchs = _search_in_ch_names(ch_names)

    # numpy array without ch_names
    elif isinstance(inst, np.ndarray) and ch_names is None:
        # data range between 0 and 255 and all integers?
        tchs = [
            idx
            for idx in range(inst.shape[0])
            if (inst[idx].astype(int, copy=False) == inst[idx]).all()
            and max(inst[idx]) <= 255
            and min(inst[idx]) == 0
        ]

    # For MNE raw/epochs + ch_names
    elif isinstance(inst, (BaseRaw, BaseEpochs)) and ch_names is not None:
        tchs = [
            idx
            for idx, type_ in enumerate(inst.get_channel_types())
            if type_ == "stim"
        ]
        if len(tchs) == 0:
            tchs = _search_in_ch_names(ch_names)

    # For MNE raw/epochs without ch_names
    elif isinstance(inst, (BaseRaw, BaseEpochs)) and ch_names is None:
        tchs = [
            idx
            for idx, type_ in enumerate(inst.get_channel_types())
            if type_ == "stim"
        ]
        if len(tchs) == 0:
            tchs = _search_in_ch_names(inst.ch_names)

    # For unknown data type
    elif inst is None:
        if ch_names is None:
            raise ValueError("ch_names cannot be None when inst is None.")
        tchs = _search_in_ch_names(ch_names)

    # output
    if len(tchs) == 0:
        return None
    elif len(tchs) == 1:
        return tchs[0]
    else:
        return tchs


def _search_in_ch_names(ch_names: List[str]) -> List[int]:
    """Search trigger channel by name in a list of valid names."""
    valid_trigger_ch_names = ["TRIGGER", "STI", "TRG", "CH_Event"]

    tchs = list()
    for idx, ch_name in enumerate(ch_names):
        if any(
            trigger_ch_name in ch_name
            for trigger_ch_name in valid_trigger_ch_names
        ):
            tchs.append(idx)

    return tchs


# ----------------------------------------------------------------------------
@fill_doc
def add_annotations_from_events(raw: BaseRaw) -> Tuple[BaseRaw, Annotations]:
    """
    Add annotations from events to the raw instance.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    annotations : Annotations
    """
    previous_annotations = raw.annotations
    tch = find_event_channel(inst=raw)
    events = mne.find_events(raw, stim_channel=raw.ch_names[tch])
    annotations = mne.annotations_from_events(
        events=events,
        event_desc=EVENTS_MAPPING,
        sfreq=raw.info["sfreq"],
        orig_time=previous_annotations.orig_time,
    )
    for k in range(events.shape[0]):
        idx = events[k, 2]
        annotations.duration[k] = EVENTS_DURATION_MAPPING[idx]
    raw.set_annotations(annotations + previous_annotations)

    return raw, annotations


@fill_doc
def check_events(raw: BaseRaw, recording_type: str) -> None:
    """
    Check that the recording has all the expected events.

    Calibration:
        1 x blink
        75 x rest
        75 x audio

    Resting-State:
        1 x resting-state

    NeuroFeedback:
        10 x regulation
        10 x non-regulation

    Parameters
    ----------
    %(raw)s
    recording_type : str
        One of 'calibration', 'rs', 'online'.

    Returns
    -------
    %(raw)s
    """
    check_functions = {
        "calibration": _check_events_calibration,
        "rs": _check_events_resting_state,
        "online": _check_events_neurofeedback,
    }
    _check_value(recording_type, check_functions, item_name="recording_type")
    tch = find_event_channel(inst=raw)
    events = mne.find_events(raw, stim_channel=raw.ch_names[tch])
    check_functions[recording_type](raw, events)


def _check_events_calibration(raw: BaseRaw, events: NDArray[int]) -> None:
    """
    Checks the event count and value in the calibration recordings.
    """
    # check the number of different events.
    count = Counter(events[:, 2])
    assert len(count.keys()) == 3, (
        "Calibration should include 3 different event keys. "
        f"Found {tuple(count.keys())}"
    )

    # check that the numbers of events are (1, 75, 75).
    count = sorted(count.items(), key=lambda x: (x[1], x[0]))
    assert count[0][1] == 1, (
        "Calibration should have a single event for blink paradigm."
        f"Found {count[0][1]}."
    )
    assert count[1][1] == 75, (
        "Calibration should include 75 x rest and 75 x audio. "
        f"Found for id {count[1][0]}: {count[1][1]}."
    )
    assert count[2][1] == 75, (
        "Calibration should include 75 x rest and 75 x audio. "
        f"Found for id {count[2][0]}: {count[2][1]}."
    )

    # check the value of each events.
    try:
        assert events[0, 2] == EVENTS["blink"]
    except AssertionError:
        replace_event_value(raw, events[0, 2], EVENTS["blink"])
    try:
        assert events[1, 2] == EVENTS["rest"]
    except AssertionError:
        replace_event_value(raw, events[1, 2], EVENTS["rest"])
    try:
        assert events[2, 2] == EVENTS["audio"]
    except AssertionError:
        replace_event_value(raw, events[2, 2], EVENTS["audio"])


def _check_events_resting_state(raw: BaseRaw, events: NDArray[int]) -> None:
    """
    Checks the event count and value in the resting-state recordings.
    """
    # check count
    assert events.shape[0] == 1, (
        "Resting-State files should have only one event. "
        f"Found {events.shape[0]}."
    )

    # check value
    try:
        assert events[0, 2] == EVENTS["resting-state"]
    except AssertionError:
        replace_event_value(raw, events[0, 2], EVENTS["resting-state"])


def _check_events_neurofeedback(raw: BaseRaw, events: NDArray[int]) -> None:
    """
    Checks the event count and value in the neurofeedback recordings.
    """
    # check the number of different events.
    count = Counter(events[:, 2])
    assert len(count.keys()) == 2, (
        "Neurofeedback should include 2 different event keys. "
        f"Found {tuple(count.keys())}"
    )

    # check that the numbers of events are (10, 10).
    count = sorted(count.items(), key=lambda x: (x[1], x[0]))
    assert count[0][1] == 10, (
        "Neurofeedback should include 10 x regulation and 10 x "
        f"non-regulation. Found for id {count[0][0]}: {count[0][1]}."
    )
    assert count[1][1] == 10, (
        "Neurofeedback should include 10 x regulation and 10 x "
        f"non-regulation. Found for id {count[1][0]}: {count[1][1]}."
    )

    # Check the value of each events.
    try:
        assert events[0, 2] == EVENTS["non-regulation"]
    except AssertionError:
        replace_event_value(raw, count[0][0], EVENTS["non-regulation"])
    try:
        assert events[1, 2] == EVENTS["regulation"]
    except AssertionError:
        replace_event_value(raw, events[1, 2], EVENTS["regulation"])


@fill_doc
def replace_event_value(
    raw: BaseRaw, old_value: int, new_value: int
) -> BaseRaw:
    """
    Replace an event value on the trigger channel.

    Parameters
    ----------
    %(raw)s
    old_value : int
        Event value to replace.
    new_value : int
        Event value replacing the old one.

    Returns
    -------
    %(raw)s
    """
    tch = find_event_channel(inst=raw)
    raw.apply_function(
        _replace_event_values_arr,
        old_value=old_value,
        new_value=new_value,
        picks=raw.ch_names[tch],
        channel_wise=True,
    )
    return raw


def _replace_event_values_arr(
    timearr: NDArray[float], old_value: int, new_value: int
) -> NDArray[int]:
    """
    Replace the values 'old_value' with 'new_value' for the array timearr.
    """
    timearr[np.where(timearr == old_value)] = new_value
    return timearr
