import mne
from mne.time_frequency import psd_welch, psd_multitaper

from utils import make_epochs


def _compute_psd(raw, method='welch', **kwargs):
    """
    Compute the power spectral density on the regulation and non-regulation
    phase of the raw instance.
    """
    method = _check_method(method)
    epochs = make_epochs(raw)

    psds, freqs = dict(), dict()
    if method == 'welch':
        for phase in epochs:
            psds[phase], freqs[phase] = psd_welch(epochs[phase])
    elif method == 'multitaper':
        for phase in epochs:
            psds[phase], freqs[phase] = psd_multitaper(epochs[phase])

    return psds, freqs


def _check_method(method):
    """Check argument method."""
    method = method.lower().strip()
    assert method in ('welch', 'multitaper'), 'Supported: welch, multitaper.'
    return method
