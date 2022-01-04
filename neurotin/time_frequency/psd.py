import re
import traceback
import multiprocessing as mp

import mne
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.integrate import simpson
from mne.time_frequency import psd_welch

from .epochs import make_fixed_length_epochs, reject_epochs
from .. import logger
from ..io.list_files import list_raw_fif
from ..io.model import load_session_weights
from ..utils.docs import fill_doc
from ..utils.checks import (_check_path, _check_participants, _check_type,
                            _check_value, _check_n_jobs)

mne.set_log_level('WARNING')


@fill_doc
def compute_psd_average_bins(folder, participants, duration, overlap, reject,
                             fmin, fmax, average='mean', n_jobs=1, **kwargs):
    """
    Compute the PSD and average by frequency band for the given participants
    using the welch method.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing preprocessed files.
    participants : int | list | tuple
        Participant ID or list of participant IDs to analyze.
    %(psd_duration)s
    %(psd_overlap)s
    %(psd_reject)s
    fmin : int | float
        Min frequency of interest.
    fmax : int | float
        Max frequency of interest.
    average : 'mean' | 'integrate'
        How to average the frequency bin/spectrum. Either 'mean' to calculate
        the arithmetic mean of all bins or 'integrate' to use Simpson's rule to
        compute integral from samples.
    %(n_jobs)s
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
    _check_type(average, (str, ), item_name='average')
    _check_value(average, ('mean', 'integrate'), item_name='average')
    n_jobs = _check_n_jobs(n_jobs)

    # create input_pool
    input_pool = list()
    for participant in participants:
        fnames = list_raw_fif(folder / str(participant).zfill(3))
        for fname in fnames:
            if fname.parent.name != 'Online':
                continue
            input_pool.append((participant, fname, duration, overlap, reject,
                               fmin, fmax, average))
    assert 0 < len(input_pool)  # sanity check

    # compute psds
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_compute_psd_average_bins, input_pool)

    # construct dataframe
    psd_dict = dict()
    for participant, session, run, psds, ch_names in results:
        for phase in psds:
            _add_data_to_dict(psd_dict, participant, session, run,
                              phase, psds[phase], ch_names)
    df = pd.DataFrame.from_dict(psd_dict, orient='columns')

    return df


def _compute_psd_average_bins(participant, fname, duration, overlap, reject,
                              fmin, fmax, average):
    """
    Compute the PSD and average by frequency band for the given participants
    using the welch method.
    """
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
        psds, freqs = _compute_psd_welch(raw, duration, overlap,
                                         reject, fmin=fmin, fmax=fmax)
        # find channel names
        ch_names = raw.pick_types(eeg=True, exclude=[]).ch_names
        assert len(ch_names) == 64  # sanity check

        psds_ = dict()
        for phase in psds:
            if average == 'mean':
                psds_[phase] = np.average(psds[phase], axis=-1)
            elif average == 'integrate':
                psds_[phase] = simpson(psds[phase], freqs[phase], axis=-1)
            assert psds_[phase].shape == (64, )  # sanity check

        # clean up
        del raw

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())

    return participant, session, run, psds_, ch_names


def _compute_psd_welch(raw, duration, overlap, reject, **kwargs):
    """
    Compute the power spectral density on the regulation and non-regulation
    phase of the raw instance using welch method.
    """
    epochs = make_fixed_length_epochs(raw, duration, overlap)
    kwargs = _check_kwargs(kwargs, epochs)
    epochs, reject = reject_epochs(epochs, reject)

    psds, freqs = dict(), dict()
    for phase in epochs.event_id:
        psds[phase], freqs[phase] = psd_welch(epochs[phase], **kwargs)
        psds[phase] = np.average(psds[phase], axis=0)

    return psds, freqs


def _check_kwargs(kwargs, epochs):
    """Check kwargs provided to _compute_psd_welch."""
    if 'picks' not in kwargs:
        kwargs['picks'] = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if 'n_fft' not in kwargs:
        kwargs['n_fft'] = epochs._data.shape[-1]
        logger.debug("Argument 'n_fft' set to %i", epochs._data.shape[-1])
    else:
        logger.warning("Argument 'n_fft' was provided and is set to %i",
                       kwargs['n_fft'])
    if 'n_overlap' not in kwargs:
        kwargs['n_overlap'] = 0
        logger.debug("Argument 'n_overlap' set to 0")
    else:
        logger.warning("Argument 'n_overlap' was provided and is set to %i",
                       kwargs['n_overlap'])
    if 'n_per_seg' not in kwargs:
        kwargs['n_per_seg'] = epochs._data.shape[-1]
        logger.debug("Argument 'n_per_seg' set to %i", epochs._data.shape[-1])
    else:
        logger.warning("Argument 'n_per_seg' was provided and is set to %i",
                       kwargs['n_per_seg'])
    return kwargs


def _add_data_to_dict(data_dict, participant, session, run, phase, data,
                      ch_names):
    """Add PSD to data dictionary."""
    keys = ['participant', 'session', 'run', 'phase', 'idx'] + ch_names

    # init
    for key in keys:
        if key not in data_dict:
            data_dict[key] = list()

    # fill data
    data_dict['participant'].append(participant)
    data_dict['session'].append(session)
    data_dict['run'].append(run)
    data_dict['phase'].append(phase[:-2])
    data_dict['idx'].append(int(phase[-1]))  # idx of the phase within the run

    # channel psd
    for k in range(data.shape[0]):
        data_dict[f'{ch_names[k]}'].append(data[k])

    # sanity check
    entries = len(data_dict['participant'])
    assert all(len(data_dict[key]) == entries for key in keys)


@fill_doc
def apply_weights_session(df, raw_folder, *, copy=False):
    """
    Apply the weights used during a given session to the PSD dataframe.

    Parameters
    ----------
    %(psd_df)s
    %(raw_folder)s
    %(copy)s

    Returns
    -------
    %(psd_df)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    raw_folder = _check_path(raw_folder, item_name='raw_folder',
                             must_exist=True)
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df

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
def apply_weights_mask(df, weights, *, copy=False):
    """
    Apply the weights mask to the PSD dataframe.

    Parameters
    ----------
    %(psd_df)s
    %(df_weights)s
    %(copy)s

    Returns
    -------
    %(psd_df)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    _check_type(weights, (pd.DataFrame, ), item_name='weights')
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df

    ch_names = weights['channel']
    df.loc[:, ch_names] = df[ch_names] * weights['weight'].values
    return df


@fill_doc
def add_average_column(df, *, copy=False):
    """
    Add a column averaging the power on all channels.

    Parameters
    ----------
    %(psd_df)s
        An 'avg' column is added averaging the power on all channels.
    %(copy)s

    Returns
    -------
    %(psd_df)s
        The average power across channels has been added in the column 'avg'.
    """
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df

    ch_names = [
        col for col in df.columns
        if col not in ('participant', 'session', 'run', 'phase', 'idx')]
    df['avg'] = df[ch_names].mean(axis=1)
    return df


@fill_doc
def remove_outliers(df, score=2., *, copy=False):
    """
    Remove outliers from the average column.

    Parameters
    ----------
    %(psd_df)s
        An 'avg' column is added averaging the power on all channels if it is
        not present.
    score : float
        ZScore threshold applied on each participant/session/run to eliminate
        outliers.
    %(copy)s

    Returns
    -------
    %(psd_df)s
        Outliers have been removed.
    """
    _check_type(score, ('numeric', ), item_name='score')
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df
    if 'avg' not in df.columns:
        df = add_average_column(df)

    outliers_idx = list()
    participants = sorted(df['participant'].unique())
    for participant in participants:
        df_participant = df[df['participant'] == participant]

        sessions = sorted(df_participant['session'].unique())
        for session in sessions:
            df_session = df_participant[df_participant['session'] == session]

            runs = sorted(df_session['run'].unique())
            for run in runs:
                df_run = df_session[df_session['run'] == run]

                # search for outliers and retrieve index
                outliers = df_run[~(np.abs(zscore(df_run['avg'])) <= score)]
                outliers_idx.extend(list(outliers.index))

    df.drop(index=outliers_idx, inplace=True)
    return df


@fill_doc
def diff_between_phases(df):
    """
    Compute the difference between the PSD in a regulation phase and in the
    preceding non-regulation phase.

    Parameters
    ----------
    %(psd_df)s
        An 'avg' column is added averaging the power on all channels if it is
        not present.

    Returns
    -------
    %(psd_diff_df)s
    """
    if 'avg' not in df.columns:
        df = add_average_column(df)

    # check keys
    keys = ['participant', 'session', 'run', 'idx']
    assert len(set(keys).intersection(df.columns)) == len(keys)

    # container for new df with diff between phases
    data = {key: [] for key in keys + ['diff']}

    participants = sorted(df['participant'].unique())
    for participant in participants:
        df_participant = df[df['participant'] == participant]

        sessions = sorted(df_participant['session'].unique())
        for session in sessions:
            df_session = df_participant[df_participant['session'] == session]

            runs = sorted(df_session['run'].unique())
            for run in runs:
                df_run = df_session[df_session['run'] == run]

                index = sorted(df_session['idx'].unique())
                for idx in index:
                    df_idx = df_run[df_run['idx'] == idx]

                    # compute the difference between regulation and rest
                    reg = df_idx[df_idx.phase == 'regulation']['avg']
                    non_reg = df_idx[df_idx.phase == 'non-regulation']['avg']
                    try:
                        diff = reg.values[0] - non_reg.values[0]
                    except IndexError:
                        continue

                    # fill dict
                    data['participant'].append(participant)
                    data['session'].append(session)
                    data['run'].append(run)
                    data['idx'].append(idx)
                    data['diff'].append(diff)

    # create df
    df = pd.DataFrame.from_dict(data, orient='columns')
    return df