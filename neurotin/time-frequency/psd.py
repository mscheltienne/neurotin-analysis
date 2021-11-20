import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mne.time_frequency import psd_welch, psd_multitaper

from utils import make_epochs
from ..ui.list_files import list_raw_fif

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
        picks_reg = mne.pick_types(epochs['regulation'].info,
                                   eeg=True, exclude=[])
        picks_rest = mne.pick_types(epochs['non-regulation'].info,
                                    eeg=True, exclude=[])
        assert (picks_reg == picks_rest).all()  # sanity check
        kwargs['picks'] = picks_reg

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


def compute_psd_average_bins(folder, participants, method='welch', **kwargs):
    """
    Compute the PSD and average by frequency bin for the given participants.
    Takes about 7.45 s ± 323 ms / session on MacBook.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing preprocessed files.
    participants : int | list of int
        Participant ID or list of participant IDs to analyze.
    method : str
        Either 'welch' for psd_welch or 'multitaper' for psd_multitaper.
    **kwargs : dict
        kwargs are passed to MNE PSD function.

    Returns
    -------
    df : DataFrame
        PSD in alpha and delta band averaged by bin and channels. Columns:
            participant : int - Participant ID
            session : int - Session ID (1 to 15)
            run : int - Run ID
            phase : str - 'regulation' or 'non-regulation'
            idx : ID of the phase within the run (1 to 10)
            alpha_ch : float - Averaged alpha PSD (1 per channel).
            delta_ch : float - Averaged delta PSD (1 per channel).
    """
    folder = _check_folder(folder)
    participants = _check_participants(participants)

    data = dict()
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

                # find ch_names
                ch_names = raw.pick_types(eeg=True, exclude=[]).ch_names
                assert len(ch_names) == 64  # sanity check

                for phase in ('regulation', 'non-regulation'):
                    alpha = np.average(psds_alpha[phase], axis=2)
                    delta = np.average(psds_delta[phase], axis=2)
                    # sanity check
                    assert alpha.shape == delta.shape == (10, 64)

                    _add_data_to_dict(data, participant, session, run, phase,
                                      alpha, delta, ch_names)
            except Exception:
                print(f'Skipping {fname}..')

    df = pd.DataFrame.from_dict(data, orient='columns')

    return df


def _add_data_to_dict(data, participant, session, run, phase, alpha, delta,
                      ch_names):
    """Add PSD to data dictionary."""
    keys = ['participant', 'session', 'run', 'phase', 'idx'] + \
           [f'alpha_{ch}' for ch in ch_names] + \
           [f'delta_{ch}' for ch in ch_names]

    # init
    for key in keys:
        if key not in data:
            data[key] = list()

    # fill data
    for k in range(alpha.shape[0]):
        data['participant'].append(participant)
        data['session'].append(session)
        data['run'].append(run)
        data['phase'].append(phase)
        data['idx'].append(k+1)  # idx of the phase within the run

        # channel psd
        for j in range(alpha.shape[1]):
            data[f'alpha_{ch_names[j]}'].append(alpha[k, j])
            data[f'delta_{ch_names[j]}'].append(delta[k, j])

    # sanity check
    entries = len(data['participant'])
    assert all(len(data[key]) == entries for key in keys)


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
