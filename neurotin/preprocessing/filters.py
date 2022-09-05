from typing import Any, Tuple

import numpy as np
from mne.io import BaseRaw

from ..utils._checks import _check_type
from ..utils._docs import fill_doc


def _check_bandpass(bandpass: Any) -> Tuple[float, float]:
    """Check that the argument bandpass is a 2-length valid list-like."""
    _check_type(bandpass, (np.ndarray, tuple, list), item_name="bandpass")
    if isinstance(bandpass, np.ndarray):
        assert bandpass.ndim == 1
    bandpass = tuple([elt if elt is None else float(elt) for elt in bandpass])
    assert len(bandpass) == 2
    assert all(0 < fq for fq in bandpass if fq is not None)
    return bandpass


def _apply_bandpass_filter(raw: BaseRaw, bandpass, picks) -> None:
    """Apply a bandpass FIR acausal filter."""
    raw.filter(
        l_freq=bandpass[0],
        h_freq=bandpass[1],
        picks=picks,
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )


def _apply_notch_filter(raw: BaseRaw, picks) -> None:
    """Filter the EU powerline noise at (50, 100, 150) Hz with a notch."""
    raw.notch_filter(np.arange(50, 151, 50), picks=picks)


@fill_doc
def apply_filter_eeg(
    raw: BaseRaw,
    *,
    bandpass=(None, None),
    notch: bool = False,
) -> None:
    """Apply filters in-place to the EEG channels.

    Parameters
    ----------
    %(raw)s
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass or highpass filter can be disabled by using None.
    notch : bool
        If True, a notch filter at (50, 100, 150) Hz  is applied, removing EU
        powerline noise.
    """
    _check_type(raw, (BaseRaw,), "raw")
    bandpass = _check_bandpass(bandpass)
    _check_type(notch, (bool,), item_name="notch")

    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, "eeg")

    if notch:
        _apply_notch_filter(raw, "eeg")


@fill_doc
def apply_filter_aux(
    raw: BaseRaw, *, bandpass=(None, None), notch: bool = False
) -> None:
    """Apply filters in-place to the AUX channels.

    Parameters
    ----------
    %(raw)s
    bandpass : tuple
        A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
        The lowpass or highpass filter can be disabled by using None.
    notch : bool
        If True, a notch filter at (50, 100, 150) Hz  is applied, removing EU
        powerline noise.
    """
    _check_type(raw, (BaseRaw,), "raw")
    bandpass = _check_bandpass(bandpass)
    _check_type(notch, (bool,), item_name="notch")

    if not all(bp is None for bp in bandpass):
        _apply_bandpass_filter(raw, bandpass, ["eog", "ecg"])

    if notch:
        _apply_notch_filter(raw, ["eog", "ecg"])
