import mne


EVENTS = {
    'rest': 1,
    'blink': 2,
    'resting-state': 3,
    'audio': 4,
    'regulation': 5,
    'non-regulation': 6
    }
EVENTS_MAPPING = {
    1: 'rest',
    2: 'blink',
    3: 'resting-state',
    4: 'audio',
    5: 'regulation',
    6: 'non-regulation'
}
EVENTS_DURATION_MAPPING = {
    1: 1,
    2: 60,
    3: 120,
    4: 0.8,
    5: 16,
    6: 8
}


def add_annotations_from_events(raw):
    """
    Add annotations from events to the raw instance.

    Parameters
    ----------
    raw : Raw
        Raw instance to modify.

    Returns
    -------
    raw : Raw instance modified in-place.
    annotations : Annotations
    """
    previous_annotations = raw.annotations
    events = mne.find_events(raw, stim_channel='TRIGGER')
    annotations = mne.annotations_from_events(
        events=events, event_desc=EVENTS_MAPPING, sfreq=raw.info['sfreq'])
    for k in range(events.shape[0]):
        idx = events[k, 2]
        annotations.duration[k] = EVENTS_DURATION_MAPPING[idx]
    raw.set_annotations(annotations + previous_annotations)

    return raw, annotations


def replace_event_value(raw, old_value, new_value):
    """
    Replace the an event value on the trigger channel.

    Parameters
    ----------
    raw : Raw
        Raw instance to modify.
    old_value : int
        Event value to replace.
    new_value : int
        Event value replacing the old one.

    Returns
    -------
    raw : Raw instance modified in-place
    """
    pass
