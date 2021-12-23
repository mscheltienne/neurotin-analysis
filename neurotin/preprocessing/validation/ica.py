"""Validation (slow) test to check different ICA methods, ocular components and
heartbeat components detection methods and thresholds."""

import os
import random
import pickle
import traceback
import multiprocessing as mp
from collections import Counter

import mne
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ... import logger
from ...io.list_files import raw_fif_selection, list_ica_fif
from ...utils.docs import fill_doc
from ...utils.checks import _check_path, _check_n_jobs


# -----------------------------------------------------------------------------
# Global test settings
FIND_ON_RAW = True  # bool
FIND_ON_EPOCHS = True  # bool
ICA_KWARGS = dict(method='picard', max_iter='auto')  # dict
FIND_BADS_EOG_KWARGS = [
    dict(measure='zscore', threshold=3.0),
    dict(measure='zscore', threshold=3.2),
    dict(measure='zscore', threshold=3.4),
    dict(measure='zscore', threshold=3.6),
    dict(measure='zscore', threshold=3.8),
    dict(measure='zscore', threshold=4.0),
    dict(measure='zscore', threshold=4.2),
    dict(measure='zscore', threshold=4.4),
    dict(measure='zscore', threshold=4.6),
    dict(measure='zscore', threshold=4.8),
    dict(measure='zscore', threshold=5.0),
    dict(measure='zscore', threshold=5.5),
    dict(measure='zscore', threshold=6.0),
    dict(measure='zscore', threshold=7.0),
    dict(measure='correlation', threshold=0.5),
    dict(measure='correlation', threshold=0.6),
    dict(measure='correlation', threshold=0.7),
    dict(measure='correlation', threshold=0.75),
    dict(measure='correlation', threshold=0.8),
    dict(measure='correlation', threshold=0.85),
    dict(measure='correlation', threshold=0.9),
]
FIND_BADS_ECG_KWARGS = [
    dict(method='correlation', measure='zscore', threshold=3.0),
    dict(method='correlation', measure='zscore', threshold=4.0),
    dict(method='correlation', measure='zscore', threshold=4.5),
    dict(method='correlation', measure='zscore', threshold=5.0),
    dict(method='correlation', measure='zscore', threshold=5.2),
    dict(method='correlation', measure='zscore', threshold=5.4),
    dict(method='correlation', measure='zscore', threshold=5.6),
    dict(method='correlation', measure='zscore', threshold=5.8),
    dict(method='correlation', measure='zscore', threshold=6.0),
    dict(method='correlation', measure='zscore', threshold=6.2),
    dict(method='correlation', measure='zscore', threshold=6.4),
    dict(method='correlation', measure='zscore', threshold=6.6),
    dict(method='correlation', measure='zscore', threshold=6.8),
    dict(method='correlation', measure='zscore', threshold=7.0),
    dict(method='correlation', measure='correlation', threshold=0.5),
    dict(method='correlation', measure='correlation', threshold=0.6),
    dict(method='correlation', measure='correlation', threshold=0.7),
    dict(method='correlation', measure='correlation', threshold=0.75),
    dict(method='correlation', measure='correlation', threshold=0.8),
    dict(method='correlation', measure='correlation', threshold=0.85),
    dict(method='correlation', measure='correlation', threshold=0.9),
]

# Checks
assert FIND_ON_EPOCHS or FIND_ON_RAW
assert isinstance(ICA_KWARGS, dict)
if isinstance(FIND_BADS_EOG_KWARGS, dict):
    FIND_BADS_EOG_KWARGS = [FIND_BADS_EOG_KWARGS]
if isinstance(FIND_BADS_ECG_KWARGS, dict):
    FIND_BADS_ECG_KWARGS = [FIND_BADS_ECG_KWARGS]


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname):
    """%(pipeline_header)s

    Compute and apply ICA-based ocular and heartbeat artifact rejection with
    the different ICA and detection methods.

    Parameters
    ----------
    %(fname)s

    Returns
    -------
    %(success)s
    %(fname)s
    eog_scores : dict
        Correlation scores for EOG related components. Keys are automatically
        generated based on the method used to select the components.
    ecg_scores : dict
        Correlation scores for ECG related components. Keys are automatically
        generated based on the method used to select the components.
    """
    logger.info('Processing: %s' % fname)
    try:
        raw = mne.io.read_raw_fif(fname, preload=True)

        if FIND_ON_EPOCHS:
            eog_epochs = mne.preprocessing.create_eog_epochs(
                raw, ch_name='EOG', picks=['eeg', 'eog'])
            ecg_epochs = mne.preprocessing.create_ecg_epochs(
                raw, ch_name='ECG', picks=['eeg', 'ecg'])

        # Fit
        ica = mne.preprocessing.ICA(**ICA_KWARGS)
        ica.fit(raw, picks='eeg', reject_by_annotation=True)

        # Init results
        eog_results_scores = dict()
        ecg_results_scores = dict()

        # Find EOG-related components
        for kwargs in FIND_BADS_EOG_KWARGS:
            if FIND_ON_RAW:
                eog_idx, eog_scores = ica.find_bads_eog(raw, **kwargs)
                key = _create_key(ICA_KWARGS, kwargs, type_='eog', on_raw=True)
                eog_results_scores[key] = eog_scores[eog_idx]
            if FIND_ON_EPOCHS:
                eog_idx, eog_scores = ica.find_bads_eog(eog_epochs, **kwargs)
                key = _create_key(ICA_KWARGS, kwargs, type_='eog', on_epo=True)
                eog_results_scores[key] = eog_scores[eog_idx]

        # Find ECG-related components
        for kwargs in FIND_BADS_ECG_KWARGS:
            if FIND_ON_RAW:
                ecg_idx, ecg_scores = ica.find_bads_ecg(raw, **kwargs)
                key = _create_key(ICA_KWARGS, kwargs, type_='ecg', on_raw=True)
                ecg_results_scores[key] = ecg_scores[ecg_idx]
            if FIND_ON_EPOCHS:
                ecg_idx, ecg_scores = ica.find_bads_ecg(ecg_epochs, **kwargs)
                key = _create_key(ICA_KWARGS, kwargs, type_='ecg', on_epo=True)
                ecg_results_scores[key] = ecg_scores[ecg_idx]

        return (True, str(fname), eog_results_scores, ecg_results_scores)

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname), dict(), dict())


def _create_key(ica_kwargs, find_bads_kwargs, type_,
                on_raw=False, on_epo=False):
    """Create a clean str key from the provided kwargs."""
    ica_kwargs_repr = \
        str(ica_kwargs).replace(
            '{', '(').replace('}', ')').replace("'", "")
    find_bads_kwargs_repr = \
        str(find_bads_kwargs).replace(
            '{', '(').replace('}', ')').replace("'", "")
    if on_raw and not on_epo:
        dtype = 'Raw'
    elif not on_raw and on_epo:
        dtype = 'Epochs'
    else:
        raise ValueError('Must be either find on Raw or on Epochs.')

    if type_ == 'eog':
        repr_ = f'EOG - {dtype} - {ica_kwargs_repr} - {find_bads_kwargs_repr}'
    elif type_ == 'ecg':
        repr_ = f'ECG - {dtype} - {ica_kwargs_repr} - {find_bads_kwargs_repr}'
    else:
        raise ValueError('Must be either eog or ecg.')

    return repr_


@fill_doc
def _cli(input_dir_fif, result_file, n_jobs=1, participant=None, session=None,
         fname=None):
    """
    Load all preprocessed raws in folder (and subfolders) sequentially
    and retrieves the number of excluded EOG and ECG components and their
    associated scores.

    Parameters
    ----------
    %(input_dir_fif)s
    result_file : str | Path
        Path to the result file in which the test results are pickled.
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    """
    input_dir_fif = _check_path(input_dir_fif, item_name='input_dir_fif',
                                must_exist=True)
    result_file = _check_result_file(result_file)
    n_jobs = _check_n_jobs(n_jobs)

    # list files to process
    fifs_in = raw_fif_selection(input_dir_fif, input_dir_fif,
                                participant=participant, session=session,
                                fname=fname, ignore_existing=False)
    input_pool = [(fname, ) for fname in fifs_in]
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    with open(result_file, mode='wb') as f:
        pickle.dump(results, f, -1)


def _check_result_file(result_file):
    """Checks that the result_file exists and is a pathlib.Path instance."""
    result_file = _check_path(result_file, item_name='result_file')
    os.makedirs(result_file.parent, exist_ok=True)
    with open(result_file, mode='w') as f:
        f.write('data will be written here..')
    return result_file


# -----------------------------------------------------------------------------
def plot_results(result_file, swarmplot=False, title_mapping=None,
                 key_to_plot='all'):
    """
    Box plot showing the repartition of the results.

    Parameters
    ----------
    result_file : str | Path
        Path to the result file in which the test results are pickled.
    swarmplot : bool
        If True, a swarmplot is overlayed on top of the boxplots.
    title_mapping : dict
        Dictionary to map the keys with the desired titles.
    key_to_plot : str | list of str | 'all'
        Subset of keys to plot. If 'all', plot all keys in result_file.
    """
    title_mapping = dict() if title_mapping is None else title_mapping
    key_to_plot = _check_key_to_plot(key_to_plot)
    results = _result_file_parser(result_file)
    for df, counter, key in results:
        if len(df) == 0:
            continue  # skip, nothing to plot.
        if key_to_plot != 'all' and key not in key_to_plot:
            continue  # skip
        title = title_mapping[key] if key in title_mapping else key
        _plot_distribution(df, counter, title, swarmplot=swarmplot, ax=None)


def _check_key_to_plot(key_to_plot):
    """Check argument key_to_plot."""
    if key_to_plot == 'all':
        return key_to_plot
    if isinstance(key_to_plot, str):
        key_to_plot = [key_to_plot]
    elif isinstance(key_to_plot, (tuple, list)):
        key_to_plot = list(key_to_plot)
        assert all(isinstance(key, str) for key in key_to_plot)
    else:
        raise ValueError('Argument key_to_plot is not valid.')
    return key_to_plot


def _result_file_parser(result_file):
    """Parse the result file and returns dataframes."""
    with open(result_file, mode='rb') as f:
        data = pickle.load(f)
    data = [elt for elt in data if elt[0]]  # Remove fails.

    # list keys
    keys = list()
    if FIND_ON_RAW:
        keys.extend([_create_key(ICA_KWARGS, kwargs, type_='eog', on_raw=True)
                     for kwargs in FIND_BADS_EOG_KWARGS])
        keys.extend([_create_key(ICA_KWARGS, kwargs, type_='ecg', on_raw=True)
                     for kwargs in FIND_BADS_ECG_KWARGS])
    if FIND_ON_RAW:
        keys.extend([_create_key(ICA_KWARGS, kwargs, type_='eog', on_epo=True)
                     for kwargs in FIND_BADS_EOG_KWARGS])
        keys.extend([_create_key(ICA_KWARGS, kwargs, type_='ecg', on_epo=True)
                     for kwargs in FIND_BADS_ECG_KWARGS])
    keys = sorted(keys)
    assert sorted(list(data[2][2]) + list(data[2][3])) == keys

    # parse
    data_per_key = {key: [] for key in keys}
    for elt in data:
        for key in keys:
            if key.startswith('EOG'):
                for j, score in enumerate(elt[2][key]):
                    data_per_key[key].append((elt[2][key].shape[0],
                                              j, abs(score)))
            elif key.startswith('ECG'):
                for j, score in enumerate(elt[3][key]):
                    data_per_key[key].append((elt[3][key].shape[0],
                                              j, abs(score)))
            else:
                raise ValueError('Key should start with EOG or ECG.')
    del data

    # clean-up
    for key, data in data_per_key.items():
        data_per_key[key] = [elt for elt in data if elt[0] <= 3]

    # counters
    counters = {key: Counter(elt[0] for elt in data)
                for key, data in data_per_key.items()}

    # patch together output
    output = [
        (pd.DataFrame(data_per_key[key],
                      columns=['n_total_comp', 'idx_comp', 'score']),
         counters[key],
         key)
        for key in keys
    ]

    return output


def _plot_distribution(df, counter, title, swarmplot=False, ax=None):
    """Plot the score distribution on an axis."""
    if ax is None:
        f, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 5))

    # plots
    sns.boxplot(x='n_total_comp', y='score', hue='idx_comp',
                data=df, ax=ax)
    # (optional) swarmplot
    if bool(swarmplot):
        sns.swarmplot(x='n_total_comp', y='score', hue='idx_comp',
                      dodge=True, data=df, ax=ax, size=1.5, color=".2")

    # figure settings
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Normalized scores')
    ax.set_xlabel('Number of total components')
    ax.set_ylim([0., 1.1])  # add headroom for text
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.legend().set_visible(False)

    # add counter as text
    for k in range(3):
        n = int(counter[k+1]/(k+1))
        if n != 0:
            ax.text(x=k, y=1.05, s='n=%i' % n, ha='center', va='center')


# -----------------------------------------------------------------------------
def random_plot_sources(prepare_raw_dir, ica_raw_dir):
    """
    Randomly pick a processed file and plot the sources (included and excluded)
    to confirm that the excluded sources are occular or heartbeat related
    components.

    Parameters
    ----------
    prepare_raw_dir : str | Path
        Path to the folder containing the FIF files processed used to fit the
        ICA.
    ica_raw_dir : str | Path
        Path to the folder containg the FIF files processed on which the ICA
        has been applied. Contains both the resulting raw instance and the
        fitted ICA instance.
    """
    # mne.viz.set_browser_backend('pyqtgraph')

    prepare_raw_dir = _check_path(prepare_raw_dir, item_name='prepare_raw_dir',
                                  must_exist=True)
    ica_raw_dir = _check_path(ica_raw_dir, item_name='ica_raw_dir',
                              must_exist=True)

    logger.info('Listing files..')
    ica_files = list_ica_fif(ica_raw_dir)
    logger.info('Listing complete.')

    # Infinite loop
    while True:
        ica, raw = _select_and_load_files(prepare_raw_dir, ica_raw_dir,
                                          ica_files)
        logger.info(raw)
        ica.plot_sources(raw, show=True, block=True)


def _select_and_load_files(prepare_raw_dir, ica_raw_dir, ica_files):
    """Select and load ICA/Raw files."""
    ica_file = random.choice(ica_files)
    relative_path = ica_file.relative_to(ica_raw_dir)
    relative_path_raw = str(relative_path).replace('-ica.fif', '-raw.fif')
    raw_file = prepare_raw_dir / relative_path_raw

    ica = mne.preprocessing.read_ica(ica_file, verbose=False)
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    return ica, raw
