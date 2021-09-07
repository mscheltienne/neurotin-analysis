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
        Raw instance to modify.
    car : bool
        If True, a CAR projector is added for the EEG channels.
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass of highpass filter can be disabled by using None.
        The filter is applied on EEG, EOG and ECG channels.
    notch : str | list
        Channel type or list of channel types on which the notch filter at
        (50, 100, 150) Hz  is applied. If None, notch filtering is disabled.

    Returns
    -------
    raw : Raw instance modified in-place.
    """
    _check_bandpass(bandpass)

    # Common average reference
    if car:
        raw.set_eeg_reference(
            ref_channels="average", ch_type="eeg", projection=True)

    # Bandpass filter
    if not all(bp is None for bp in bandpass):
        raw.filter(
            l_freq=bandpass[0],
            h_freq=bandpass[1],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge")

    # Notch filters
    if notch is not None:
        raw.notch_filter(np.arange(50, 151, 50), picks=notch)

    return raw
