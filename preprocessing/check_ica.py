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
    try:
        assert Path(str(fname_out_stem) + '-raw.fif').exists()
        raw = read_raw_fif(fname)
        raw = prepare_raw(raw)
        raw.info['bads'] = list()  # bug fixed in #9719
        ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
        ica.fit(raw, picks='eeg', reject_by_annotation=True)
        eog_idx, eog_scores = ica.find_bads_eog(raw)
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw)
        return (True, str(fname), eog_scores[eog_idx], ecg_scores[ecg_idx])
    except Exception:
        print (f'FAILED: {fname}')
        print(traceback.format_exc())
        return (False, str(fname), None, None)


def main(folder_in, folder_out, output_directory, processes=1):
    """
    Load all preprocessed raws and ICA in folder (and subfolders) sequentially
    and retrieves the number of excluded EOG and ECG components and their
    associated correlation with the EOG and ECG channels.

    Parameters
    ----------
    folder_in : str | Path
        Path to the folder containing the FIF files to preprocess.
    folder_out : str | Path
        Path to the folder containing the FIF files preprocessed.
    output_directory : str | Path
        Path to the directory where the results are saved in pickle format.
    processes : int
        Number of parallel processes used.
    """
    folder_in, folder_out = _check_folders(folder_in, folder_out)
    output_directory = _check_output_directory(output_directory)
    processes = _check_processes(processes)

    raws = list_raw_fif(folder_in)
    input_pool = [(fname,
                   str(folder_out / fname.relative_to(folder_in))[:-8])
                  for fname in raws]
    with mp.Pool(processes=processes) as p:
        results = p.starmap(check_ica, input_pool)

    with open(output_directory / 'data-ica.pcl', mode='wb') as f:
        pickle.dump(results, f, -1)


def _check_folders(folder_in, folder_out):
    """Checks that the folders exist and are pathlib.Path instances."""
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)
    assert folder_in.exists(), 'The input folder does not exists.'
    assert folder_out.exists(), 'The ouput folder does not exists.'
    return folder_in, folder_out


def _check_processes(processes):
    """Checks that the number of processes is valid."""
    processes = int(processes)
    assert 0 < processes, 'processes should be a positive integer'
    return processes


def _check_output_directory(output_directory):
    """Checks that the folder exists and is a pathlib.Path instance."""
    output_directory = Path(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    with open(output_directory / 'data-ica.pcl', mode='w') as f:
        f.write('data will be written here..')
    return output_directory


def plot_results(result_file, swarmplot=False):
    """
    Box plot showing the repartition of the results.

    Parameters
    ----------
    result_file : str | Path
        Path to the result file data-ica.pcl containing the components/scores.
    """
    result_file = _check_result_file(result_file)
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


def _check_result_file(result_file):
    """
    Checks that the file exist and has the correct hardcoded name.
    """
    result_file = Path(result_file)
    assert result_file.name == 'data-ica.pcl'
    assert result_file.exists()
    return result_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin - ICA checker',
        description='Checks scores and components removed by ICA.')
    parser.add_argument(
        'folder_in', type=str,
        help='Folder containing FIF files to preprocess.')
    parser.add_argument(
        'folder_out', type=str,
        help='Folder containing FIF files preprocessed.')
    parser.add_argument(
        'output_directory', type=str,
        help='Folder where the results are pickled.')
    parser.add_argument(
        '--processes', type=int, metavar='int',
        help='Number of parallel processes.', default=1)

    args = parser.parse_args()

    main(args.folder_in, args.folder_out,
         args.output_directory, args.processes)