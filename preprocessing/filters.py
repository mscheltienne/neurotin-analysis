import numpy as np


def _check_bandpass(bandpass):
    if not isinstance(bandpass, (tuple, list)):
        raise TypeError
    if len(bandpass) != 2:
        raise ValueError
    for fq in bandpass:
        if fq is not None and fq <= 0:
            raise ValueError


def apply_filter(raw, car, bandpass, notch):
    """
    Apply filters in-place:
        - CAR
        - Bandpass
        - Notch

    Parameters
    ----------
    raw : Raw
        Raw instance modified.
    car : bool
        If True, a CAR projector is added for the EEG channels.
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass of highpass filter can be disabled by using None.
        The filter is applied on EEG, EOG and ECG channels.
    notch : bool
        If True, a notch filter at (50, 100, 150) Hz is applied on EEG, EOG
        and ECG channels.
    """
    _check_bandpass(bandpass)

    # Common average reference
    if car:
        raw.set_eeg_reference(
            ref_channels='average', ch_type='eeg', projection=True)

    # Bandpass filter
    if not all(bp is None for bp in bandpass):
        raw.filter(
            l_freq=bandpass[0], h_freq=bandpass[1],
            picks=['ecg', 'eog', 'eeg'], method='iir',
            iir_params=dict(order=4, ftype='butter', output='sos'))

    # Notch filters
    if notch and car:
        raw.notch_filter(np.arange(50, 151, 50), picks=['ecg', 'eog'])
    elif notch and not car:
        raw.notch_filter(np.arange(50, 151, 50), picks=['eeg', 'ecg', 'eog'])
