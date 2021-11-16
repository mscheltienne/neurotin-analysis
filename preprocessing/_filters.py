import numpy as np

from utils import _check_type


def _check_bandpass(bandpass):
    """
    Checks that the argument bandpass is a 2-length valid list-like.
    """
    _check_type(bandpass, (np.ndarray, tuple, list), 'bandpass')
    bandpass = tuple(np.ndarray)
    assert len(bandpass) == 2
    assert all(0 < fq for fq in bandpass if fq is not None)
    return bandpass


def _apply_bandpass_filter(raw, bandpass, picks):
    """
    Apply a bandpass FIR acausal filter.
    """
    bandpass = _check_bandpass(bandpass)
    raw.filter(
        l_freq=bandpass[0],
        h_freq=bandpass[1],
        picks=picks,
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge")


def _apply_notch_filter(raw, picks):
    """
    Filter the EU powerline noise at (50, 100, 150) Hz with a notch filter.
    """
    raw.notch_filter(np.arange(50, 151, 50), picks=picks)


def _apply_car(raw, *, projection=True):
    """
    Adds a CAR projector based on the good EEG channels.
    """
    raw.set_eeg_reference(
        ref_channels="average", ch_type="eeg", projection=projection)


def apply_filter_eeg(raw, bandpass, notch, car):
    """
    Apply filters in-place to the EEG channels:
        - Bandpass
        - Notch
        - CAR

    Parameters
    ----------
    raw : Raw
        Raw instance modified in-place.
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass or highpass filter can be disabled by using None.
    notch : bool
        If True, a notch filter at (50, 100, 150) Hz  is applied.
    car : bool
        If True, a CAR projector based on the good channels is added.
    """
    if car:
        _apply_car(raw)

    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, 'eeg')

    if notch:
        _apply_notch_filter(raw, 'eeg')


def apply_filter_aux(raw, bandpass, notch):
    """
    Apply filters in-place to the AUX channels:
        - Bandpass
        - Notch

    Parameters
    ----------
    raw : Raw
        Raw instance modified in-place.
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass or highpass filter can be disabled by using None.
    notch : bool
        If True, a notch filter at (50, 100, 150) Hz  is applied.
    """
    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, ['eog', 'ecg'])

    if notch:
        _apply_notch_filter(raw, ['eog', 'ecg'])
