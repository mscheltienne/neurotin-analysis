import os
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

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
    raw_fif_file : Path
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
        return (True, fname, eog_scores[eog_idx], ecg_scores[ecg_idx])
    except Exception:
        print (f'FAILED: {fname}')
        print(traceback.format_exc())
        return (False, fname, None, None)


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
