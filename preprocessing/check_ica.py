import traceback
from pathlib import Path
import multiprocessing as mp

import mne
import seaborn as sns

from utils import list_raw_fif


def check_ica(raw_fif_file):
    """
    Function called on each raw/ica files.

    Parameters
    ----------
    raw_fif_file : Path
        Path to the input '-raw.fif' file preprocessed.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    raw_fif_file : Path
        Path to the input '-raw.fif' file preprocessed.
    eog_idx : None | list
        The indices of EOG related components, sorted by score.
    eog_scores : None | list
        The correlation scores for EOG related components.
    ecg_idx : None | list
        The indices of ECG related components, sorted by score.
    ecg_scores : None | list
        The correlation scores for ECG related components.
    """
    try:
        ica_fif_file = Path(str(raw_fif_file).split('-raw.fif')[0]+'-ica.fif')
        assert raw_fif_file.exists()
        assert ica_fif_file.exists()
        raw = mne.io.read_raw_fif(raw_fif_file)
        ica = mne.preprocessing.read_ica(ica_fif_file)

        eog_idx, eog_scores = ica.find_bads_eog(raw)
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw)

        return (True, raw_fif_file,
                eog_idx, eog_scores[eog_idx],
                ecg_idx, ecg_scores[ecg_idx])

    except AssertionError:
        print (f'FAILED - NOT FOUND: {raw_fif_file}')
        print(traceback.format_exc())
        return (False, raw_fif_file, None, None, None, None)
    except Exception:
        print (f'FAILED: {raw_fif_file}')
        print(traceback.format_exc())
        return (False, raw_fif_file, None, None, None, None)


def main(folder, processes=1):
    """
    Load all preprocessed raws and ICA in folder (and subfolders) sequentially
    and retrieves the number of excluded EOG and ECG components and their
    associated correlation with the EOG and ECG channels.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing the FIF files preprocessed.
    processes : int
        Number of parallel processes used.
    """
    folder = _check_folder(folder)
    processes = _check_processes(processes)

    raws = list_raw_fif(folder)
    with mp.Pool(processes=processes) as p:
        results = p.starmap(check_ica, raws)


def _check_folder(folder):
    """Checks that the folder exists and is a pathlib.Path instances."""
    folder = Path(folder)
    assert folder.exists(), 'The folder does not exists.'
    return folder


def _check_processes(processes):
    """Checks that the number of processes is valid."""
    processes = int(processes)
    assert 0 < processes, 'processes should be a positive integer'
    return processes
