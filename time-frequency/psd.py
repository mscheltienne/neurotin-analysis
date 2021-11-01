import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne.time_frequency import psd_welch, psd_multitaper

from utils import make_epochs, list_raw_fif

mne.set_log_level('WARNING')


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


def compute_average_psd(folder, participants, method='welch', **kwargs):
    """Compute the average channel/bin PSD for the given participants."""
    folder = _check_folder(folder)
    participants = _check_participants(participants)

    data = list()
    for participant in participants:
        fnames = list_raw_fif(folder/str(participant).zfill(3))
        for fname in fnames:
            if fname.parent.name != 'Online':
                continue

            try:
                raw = mne.io.read_raw_fif(fname, preload=True)

                # find session id
                pattern = re.compile(r'Session (\d{1,2})')
                session = re.findall(pattern, str(fname))
                assert len(session) == 1
                session = int(session[0])
                # find run id
                run = int(fname.name.split('-')[0])

                # alpha
                kwargs['fmin'], kwargs['fmax'] = 8., 13.
                psds_alpha, _ = _compute_psd(raw, method, **kwargs)
                # delta
                kwargs['fmin'], kwargs['fmax'] = 1., 4.
                psds_delta, _ = _compute_psd(raw, method, **kwargs)

                # sanity check
                assert sorted(list(psds_alpha)) == sorted(list(psds_delta)) \
                    == ['non-regulation', 'regulation']

                for phase in ('regulation', 'non-regulation'):
                    alpha = np.average(psds_alpha[phase], axis=(1, 2))
                    delta = np.average(psds_delta[phase], axis=(1, 2))
                    # sanity check
                    assert alpha.shape == delta.shape == (10, )

                    for k in range(alpha.shape[0]):
                        data.append((participant, session, run, phase, k+1,
                                     alpha[k], delta[k]))
            except:
                print (f'Skipping {fname}..')

    df = pd.DataFrame(data, columns=['participant', 'session', 'run', 'phase',
                                     'idx', 'alpha', 'delta'])

    return df


def _check_folder(folder):
    """Check argument folder."""
    folder = Path(folder)
    assert folder.exists()
    return folder


def _check_participants(participants):
    """Check argument participants."""
    if isinstance(participants, (int, float)):
        participants = [int(participants)]
    else:
        participants = [int(participant) for participant in participants]
    assert all(50 <= participant <= 150 for participant in participants)
    return participants
