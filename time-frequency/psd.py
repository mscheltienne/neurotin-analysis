import mne
import numpy as np
from mne.time_frequency import psd_welch, psd_multitaper

from utils import make_epochs


def _compute_psd(raw, method='welch', **kwargs):
    """
    Compute the power spectral density on the regulation and non-regulation
    phase of the raw instance.
    """
    method = _check_method(method)
    epochs = make_epochs(raw)

    # select all channels
    if 'picks' not in kwargs:
        info = epochs['regulation'].info
        kwargs['picks'] = mne.pick_types(info, eeg=True, exclude=[])

    psds, freqs = dict(), dict()
    if method == 'welch':
        for phase in epochs:
            psds[phase], freqs[phase] = psd_welch(epochs[phase], **kwargs)
    elif method == 'multitaper':
        for phase in epochs:
            psds[phase], freqs[phase] = psd_multitaper(epochs[phase], **kwargs)

    return psds, freqs


def _check_method(method):
    """Check argument method."""
    method = method.lower().strip()
    assert method in ('welch', 'multitaper'), 'Supported: welch, multitaper.'

    return method


def compute_alpha_psd(raw, method='welch', **kwargs):
    """Compute the alpha band PSD."""
    kwargs['fmin'] = 8.
    kwargs['fmax'] = 13.
    psds, _ = _compute_psd(raw, method, **kwargs)
    psds_average = dict()
    for phase in psds:
        psds_average[phase] = np.average(psds[phase], axis=(1, 2))

    return psds_average


def compute_delta_psd(raw, method='welch', **kwargs):
    """Compute the delta band PSD."""
    kwargs['fmin'] = 1.
    kwargs['fmax'] = 4.
    psds, _ = _compute_psd(raw, method, **kwargs)
    psds_average = dict()
    for phase in psds:
        psds_average[phase] = np.average(psds[phase], axis=(1, 2))

    return psds_average
