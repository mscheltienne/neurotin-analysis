import re
import traceback

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_welch, psd_multitaper

from .epochs import make_epochs
from .. import logger
from ..io.list_files import list_raw_fif
from ..io.model import load_session_weights
from ..utils.docs import fill_doc
from ..utils.checks import (_check_value, _check_path, _check_participants,
                            _check_type)

mne.set_log_level('WARNING')


def _compute_psd(raw, method='welch', **kwargs):
    """
    Compute the power spectral density on the regulation and non-regulation
    phase of the raw instance.
    """
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


@fill_doc
def compute_psd_average_bins(folder, participants, fmin, fmax, method='welch',
                             **kwargs):
    """
    Compute the PSD and average by frequency band for the given participants.
    Takes about 7.45 s ± 323 ms / session on MacBook.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing preprocessed files.
    participants : int | list | tuple
        Participant ID or list of participant IDs to analyze.
    fmin : int | float
        Min frequency of interest.
    fmax : int | float
        Max frequency of interest.
    method : str
        Either 'welch' for psd_welch or 'multitaper' for psd_multitaper.
    **kwargs : dict
        kwargs are passed to MNE PSD function.

    Returns
    -------
    %(psd_df)s
    """
    folder = _check_path(folder, item_name='folder', must_exist=True)
    participants = _check_participants(participants)
    _check_type(fmin, ('numeric', ), item_name='fmin')
    _check_type(fmax, ('numeric', ), item_name='fmax')
    assert 'fmin' not in kwargs and 'fmax' not in kwargs
    _check_value(method, ('welch', 'multitaper'), item_name='method')

    psd_dict = dict()
    for participant in participants:
        fnames = list_raw_fif(folder / str(participant).zfill(3))
        for fname in fnames:
            if fname.parent.name != 'Online':
                continue

            logger.info('Processing: %s' % fname)
            try:
                raw = mne.io.read_raw_fif(fname, preload=True)
                # find session id
                pattern = re.compile(r'Session (\d{1,2})')
                session = re.findall(pattern, str(fname))
                assert len(session) == 1
                session = int(session[0])
                # find run id
                run = int(fname.name.split('-')[0])
                # compute psds
                psds, _ = _compute_psd(raw, method, fmin=fmin, fmax=fmax)
                # find channel names
                ch_names = raw.pick_types(eeg=True, exclude=[]).ch_names

                # sanity check
                assert sorted(list(psds)) == ['non-regulation', 'regulation']
                assert len(ch_names) == 64

                for phase in ('regulation', 'non-regulation'):
                    psds_ = np.average(psds[phase], axis=-1)
                    assert psds_.shape == (10, 64)  # sanity check
                    _add_data_to_dict(psd_dict, participant, session, run,
                                      phase, psds_, ch_names)

                # clean up
                del raw

            except Exception:
                logger.warning('FAILED: %s -> Skip.' % fname)
                logger.debug(traceback.format_exc())

    df = pd.DataFrame.from_dict(psd_dict, orient='columns')

    return df


def _add_data_to_dict(data_dict, participant, session, run, phase, data,
                      ch_names):
    """Add PSD to data dictionary."""
    keys = ['participant', 'session', 'run', 'phase', 'idx'] + ch_names

    # init
    for key in keys:
        if key not in data_dict:
            data_dict[key] = list()

    # fill data
    for k in range(data.shape[0]):
        data_dict['participant'].append(participant)
        data_dict['session'].append(session)
        data_dict['run'].append(run)
        data_dict['phase'].append(phase)
        data_dict['idx'].append(k+1)  # idx of the phase within the run

        # channel psd
        for j in range(data.shape[1]):
            data_dict[f'{ch_names[j]}'].append(data[k, j])

    # sanity check
    entries = len(data_dict['participant'])
    assert all(len(data_dict[key]) == entries for key in keys)


@fill_doc
def apply_weights_session(df, raw_folder):
    """
    Apply the weights used during a given session to the PSD dataframe.

    Parameters
    ----------
    %(psd_df)s
    %(raw_folder)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    raw_folder = _check_path(raw_folder, item_name='raw_folder',
                             must_exist=True)

    participant_session = None
    for index, row in df.iterrows():
        # load weights if needed
        if participant_session != (row['participant'], row['session']):
            weights = load_session_weights(raw_folder, row['participant'],
                                           row['session'])
            participant_session = (row['participant'], row['session'])
            ch_names = weights['channel']

        df.loc[index, ch_names] = row[ch_names] * weights['weight'].values

    return df


@fill_doc
def apply_weights_mask(df, weights):
    """
    Apply the weights mask to the PSD dataframe.

    Parameters
    ----------
    %(psd_df)s
    %(df_weights)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    _check_type(weights, (pd.DataFrame, ), item_name='weights')

    ch_names = weights['channel']
    df.loc[:, ch_names] = df[ch_names] * weights['weight'].values

    return df
