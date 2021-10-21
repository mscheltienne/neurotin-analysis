import os
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp
from collections import Counter

import mne
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pipeline import prepare_raw
from utils import list_raw_fif, read_raw_fif


# Path to the folder containing the FIF files to preprocess.
FOLDER_IN = r'C:\Users\Mathieu\Documents\datasets\neurotin\raw'
# Path to the folder containing the FIF files preprocessed.
FOLDER_OUT = r'C:\Users\Mathieu\Documents\datasets\neurotin\clean'
# File in which the results are pickled
RESULT_FILE = r'C:\Users\Mathieu\Documents\datasets\neurotin\data-ica.pcl'

# Global test settings
FIND_ON_RAW = True  # bool
FIND_ON_EPOCHS = True  # bool
ICA_KWARGS = dict(method='picard', max_iter='auto')  # dict
FIND_BADS_EOG_KWARGS = [
    dict(measure='zscore', threshold=3.0),
    dict(measure='zscore', threshold=4.0),
    dict(measure='zscore', threshold=5.0)
]
FIND_BADS_ECG_KWARGS = [
    dict(method='correlation', measure='zscore', threshold=3.0),
    dict(method='correlation', measure='zscore', threshold=6.0)
]

# Checks
assert FIND_ON_EPOCHS or FIND_ON_RAW
assert isinstance(ICA_KWARGS, dict)
if isinstance(FIND_BADS_EOG_KWARGS, dict):
    FIND_BADS_EOG_KWARGS = [FIND_BADS_EOG_KWARGS]
if isinstance(FIND_BADS_ECG_KWARGS, dict):
    FIND_BADS_ECG_KWARGS = [FIND_BADS_ECG_KWARGS]


def check_ica(fname, fname_out_stem):
    """
    Function called on each raw/ica files.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    fname_out_stem : str | Path
        Path and naming scheme used to save -raw.fif and -ica.fif files.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : Path
        Path to the input '-raw.fif' file preprocessed.
    eog_scores : None | list
        The correlation scores for EOG related components.
    ecg_scores : None | list
        The correlation scores for ECG related components.
    """
    # To be rework when #9846 is fixed.
    try:
        assert Path(str(fname_out_stem) + '-raw.fif').exists()
        raw = read_raw_fif(fname)
        raw = prepare_raw(raw)
        raw.info['bads'] = list()  # bug fixed in #9719

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
        print ('----------------------------------------------')
        print (f'FAILED: {fname} -> Skip.')
        print(traceback.format_exc())
        print ('----------------------------------------------')
        return (False, str(fname), dict(), dict())


def _create_key(ica_kwargs, find_bads_kwargs, type_,
                on_raw=False, on_epo=False):
    """Create a clean str key from the passed kwargs."""
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
        repr_ = f'EOG - {ica_kwargs_repr} - {find_bads_kwargs_repr} - {dtype}'
    elif type_ == 'ecg':
       repr_ = f'ECG - {ica_kwargs_repr} - {find_bads_kwargs_repr} - {dtype}'
    else:
        raise ValueError('Must be either eog or ecg.')

    return repr_


def main(processes=1):
    """
    Load all preprocessed raws and ICA in folder (and subfolders) sequentially
    and retrieves the number of excluded EOG and ECG components and their
    associated correlation with the EOG and ECG channels.

    Parameters
    ----------
    processes : int
        Number of parallel processes used.
    """
    folder_in, folder_out = _check_folders(FOLDER_IN, FOLDER_OUT)
    result_file = _check_result_file(RESULT_FILE)

    raws = list_raw_fif(folder_in)
    input_pool = [(fname,
                    str(folder_out / fname.relative_to(folder_in))[:-8])
                  for fname in raws]
    with mp.Pool(processes=processes) as p:
        results = p.starmap(check_ica, input_pool)

    with open(result_file, mode='wb') as f:
        pickle.dump(results, f, -1)


def _check_folders(folder_in, folder_out):
    """Checks that the folders exist and are pathlib.Path instances."""
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)
    assert folder_in.exists(), 'The input folder does not exists.'
    assert folder_out.exists(), 'The ouput folder does not exists.'
    return folder_in, folder_out


def _check_result_file(result_file):
    """Checks that the result_file exists and is a pathlib.Path instance."""
    result_file = Path(result_file)
    os.makedirs(result_file.parent, exist_ok=True)
    with open(result_file, mode='w') as f:
        f.write('data will be written here..')
    return result_file


def plot_results(result_file, swarmplot=False):
    """
    Box plot showing the repartition of the results.

    Parameters
    ----------
    result_file : str | Path
        Path to the result file data-ica.pcl containing the components/scores.
    """
    with open(result_file, mode='rb') as f:
        data = pickle.load(f)

    # format: (success, fname, EOG scores, ECG scores)
    data = [elt for elt in data if elt[0]]  # Remove fails.
    data_eog, data_ecg = [], []
    for elt in data:
        for k, score in enumerate(elt[2]):
            data_eog.append((elt[2].shape[0], k, abs(score)))
        for k, score in enumerate(elt[3]):
            data_ecg.append((elt[3].shape[0], k, abs(score)))

    # Remove problem, for a very small number of files, there was 4, 5 or even
    # 6 component removed. -> To be fixed in the preprocessing pipeline with
    # an assertion on the number of components.
    data_eog = [elt for elt in data_eog if elt[0] <= 3]
    data_ecg = [elt for elt in data_ecg if elt[0] <= 3]
    n_eog_components = Counter(elt[0] for elt in data_eog)
    n_ecg_components = Counter(elt[0] for elt in data_ecg)

    # Convert to dataframe
    df_eog = pd.DataFrame(data_eog,
                          columns=['n_total_comp', 'idx_comp', 'score'])
    df_ecg = pd.DataFrame(data_ecg,
                          columns=['n_total_comp', 'idx_comp', 'score'])

    # Figure settings
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 7))

    # Box plot
    sns.boxplot(x='n_total_comp', y='score', hue='idx_comp',
                data=df_eog, ax=ax[0])
    sns.boxplot(x='n_total_comp', y='score', hue='idx_comp',
                data=df_ecg, ax=ax[1])

    # (optional) Swarmplot
    if bool(swarmplot):
        sns.swarmplot(x='n_total_comp', y='score', hue='idx_comp',
                    dodge=True, data=df_eog, ax=ax[0], size=1.5, color=".2")
        sns.swarmplot(x='n_total_comp', y='score', hue='idx_comp',
                    dodge=True, data=df_ecg, ax=ax[1], size=1.5, color=".2")

    # Figure settings
    ax[0].set_title('EOG-related components')
    ax[1].set_title('ECG-related components')
    ax[0].set_ylabel('Normalized scores')
    ax[1].set_ylabel('Normalized scores')
    ax[0].set_xlabel('Number of total components')
    ax[1].set_xlabel('Number of total components')
    ax[0].set_ylim([0., 1.1])  # add headroom for text
    ax[1].set_ylim([0., 1.1])  # add headroom for text
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax[0].set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax[1].set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

    # Legend settings
    handles = ax[0].legend().legendHandles[:3]
    labels = ['1', '2', '3']
    ax[0].legend().set_visible(False)
    ax[1].legend().set_visible(False)
    f.legend(handles, labels, title='Component n°', loc="right")

    # Add text with counts
    for k in range(3):
        ax[0].text(x=k, y=1.05, s='n=%i' % int(n_eog_components[k+1]/(k+1)),
                   ha='center', va='center')
        ax[1].text(x=k, y=1.05, s='n=%i' % int(n_ecg_components[k+1]/(k+1)),
                   ha='center', va='center')

    return f, ax


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
    data_per_key = dict()
    for elt in data:
        for key in keys:
            if key.startswith('EOG'):
                for j, score in enumerate(elt[2][key]):
                    data_per_key[key].append(elt[2][key].shape[0], j, abs(score))
            elif key.startswith('ECG'):
                for j, score in enumerate(elt[3][key]):
                    data_per_key[key].append(elt[3][key].shape[0], j, abs(score))
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
        (key,
         pd.DataFrame(data_per_key[key],
                      columns=['n_total_comp', 'idx_comp', 'score']),
         counters[key]) for key in keys]

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin - ICA checker',
        description='Checks scores and components removed by ICA.')
    parser.add_argument(
        '--processes', type=int, metavar='int',
        help='Number of parallel processes.', default=1)
    args = parser.parse_args()

    main(args.processes)
