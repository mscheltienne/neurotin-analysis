from collections import Counter

import mne
import numpy as np
from bsl.utils import find_event_channel

from ..utils.docs import fill_doc
from ..utils.checks import _check_value
from ..config.events import EVENTS, EVENTS_MAPPING, EVENTS_DURATION_MAPPING


@fill_doc
def add_annotations_from_events(raw):
    """
    Add annotations from events to the raw instance.

    Parameters
    ----------
    %(raw_in_place)s

    Returns
    -------
    %(raw_in_place)s
    annotations : Annotations
    """
    previous_annotations = raw.annotations
    tch = find_event_channel(inst=raw)
    events = mne.find_events(raw, stim_channel=raw.ch_names[tch])
    annotations = mne.annotations_from_events(
        events=events, event_desc=EVENTS_MAPPING, sfreq=raw.info["sfreq"],
        orig_time=previous_annotations.orig_time)
    for k in range(events.shape[0]):
        idx = events[k, 2]
        annotations.duration[k] = EVENTS_DURATION_MAPPING[idx]
    raw.set_annotations(annotations + previous_annotations)

    return raw, annotations


@fill_doc
def check_events(raw, recording_type):
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
    %(raw_in_place)s
    recording_type : str
        One of 'calibration', 'rs', 'online'.

    Returns
    -------
    %(raw_in_place)s
    """
    check_functions = {
        "calibration": _check_events_calibration,
        "rs": _check_events_resting_state,
        "online": _check_events_neurofeedback}
    _check_value(recording_type, check_functions, item_name="recording_type")
    tch = find_event_channel(inst=raw)
    events = mne.find_events(raw, stim_channel=raw.ch_names[tch])
    check_functions[recording_type](raw, events)


def _check_events_calibration(raw, events):
    """
    Checks the event count and value in the calibration recordings.
    """
    # check the number of different events.
    count = Counter(events[:, 2])
    assert len(count.keys()) == 3, (
        "Calibration should include 3 different event keys. "
        f"Found {tuple(count.keys())}")

    # check that the numbers of events are (1, 75, 75).
    count = sorted(count.items(), key=lambda x: (x[1], x[0]))
    assert count[0][1] == 1, (
        "Calibration should have a single event for blink paradigm."
        f"Found {count[0][1]}.")
    assert count[1][1] == 75, (
        "Calibration should include 75 x rest and 75 x audio. "
        f"Found for id {count[1][0]}: {count[1][1]}.")
    assert count[2][1] == 75, (
        "Calibration should include 75 x rest and 75 x audio. "
        f"Found for id {count[2][0]}: {count[2][1]}.")

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


def _check_events_resting_state(raw, events):
    """
    Checks the event count and value in the resting-state recordings.
    """
    # check count
    assert events.shape[0] == 1, (
        "Resting-State files should have only one event. "
        f"Found {events.shape[0]}.")

    # check value
    try:
        assert events[0, 2] == EVENTS["resting-state"]
    except AssertionError:
        replace_event_value(raw, events[0, 2], EVENTS["resting-state"])


def _check_events_neurofeedback(raw, events):
    """
    Checks the event count and value in the neurofeedback recordings.
    """
    # check the number of different events.
    count = Counter(events[:, 2])
    assert len(count.keys()) == 2, (
        "Neurofeedback should include 2 different event keys. "
        f"Found {tuple(count.keys())}")

    # check that the numbers of events are (10, 10).
    count = sorted(count.items(), key=lambda x: (x[1], x[0]))
    assert count[0][1] == 10, (
        "Neurofeedback should include 10 x regulation and 10 x "
        f"non-regulation. Found for id {count[0][0]}: {count[0][1]}.")
    assert count[1][1] == 10, (
        "Neurofeedback should include 10 x regulation and 10 x "
        f"non-regulation. Found for id {count[1][0]}: {count[1][1]}.")

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
def replace_event_value(raw, old_value, new_value):
    """
    Replace an event value on the trigger channel.

    Parameters
    ----------
    %(raw_in_place)s
    old_value : int
        Event value to replace.
    new_value : int
        Event value replacing the old one.

    Returns
    -------
    %(raw_in_place)s
    """
    tch = find_event_channel(inst=raw)
    raw.apply_function(
        _replace_event_values_arr,
        old_value=old_value,
        new_value=new_value,
        picks=raw.ch_names[tch],
        channel_wise=True)
    return raw


def _replace_event_values_arr(timearr, old_value, new_value):
    """
    Replace the values 'old_value' with 'new_value' for the array timearr.
    """
    timearr[np.where(timearr == old_value)] = new_value
    return timearr
