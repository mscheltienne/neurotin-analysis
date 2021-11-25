import numpy as np

from ..utils.docs import fill_doc
from ..utils.checks import _check_type


def _check_bandpass(bandpass):
    """
    Checks that the argument bandpass is a 2-length valid list-like.
    """
    _check_type(bandpass, (np.ndarray, tuple, list), item_name='bandpass')
    bandpass = tuple(bandpass)
    assert len(bandpass) == 2
    assert all(0 < fq for fq in bandpass if fq is not None)
    return bandpass


def _apply_bandpass_filter(raw, bandpass, picks):
    """
    Apply a bandpass FIR acausal filter.
    """
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


def _apply_car(raw, *, projection=False):
    """
    Adds a CAR projector based on the good EEG channels.
    """
    raw.set_eeg_reference(
        ref_channels="average", ch_type="eeg", projection=projection)


@fill_doc
def apply_filter_eeg(raw, *, bandpass=(None, None), notch=False, car=False):
    """
    Apply filters in-place to the EEG channels:
        - Bandpass
        - Notch
        - CAR

    Parameters
    ----------
    %(raw_in_place)s
    %(bandpass)s
    %(notch)s
    car : bool
        If True, a CAR reference based on the good channels is added.
    """
    bandpass = _check_bandpass(bandpass)
    notch = _check_type(notch, (bool, ), item_name='notch')
    car = _check_type(car, (bool, ), item_name='car')

    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, 'eeg')

    if notch:
        _apply_notch_filter(raw, 'eeg')

    if car:
        _apply_car(raw, projection=False)


@fill_doc
def apply_filter_aux(raw, *, bandpass=(None, None), notch=False):
    """
    Apply filters in-place to the AUX channels:
        - Bandpass
        - Notch

    Parameters
    ----------
    %(raw_in_place)s
    %(bandpass)s
    %(notch)s
    """
    bandpass = _check_bandpass(bandpass)
    notch = _check_type(notch, (bool, ), item_name='notch')

    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, ['eog', 'ecg'])

    if notch:
        _apply_notch_filter(raw, ['eog', 'ecg'])
