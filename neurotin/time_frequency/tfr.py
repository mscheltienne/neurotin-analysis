import multiprocessing as mp
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

from mne import BaseEpochs, concatenate_epochs
from mne.io import read_raw_fif
from mne.time_frequency import AverageTFR, tfr_morlet, tfr_multitaper
from numpy.typing import NDArray

from ..utils._checks import (
    _check_n_jobs,
    _check_participants,
    _check_path,
    _check_type,
    _check_value,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..utils.selection import list_runs_pp
from .epochs import make_combine_epochs


def tfr_global(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    regular_only: bool,
    transfer_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    **kwargs,
) -> Tuple[AverageTFR, AverageTFR]:
    """Compute the global TFR avering all selected subjects and sessions.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(regular_only)s
    %(transfer_only)s
    %(participants)s
    method : 'multitaper'
        TFR method used.
    **kwargs
        Extra keyword arguments are passed to the TFR method.

    Notes
    -----
    'multitaper' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
        - time_bandwith: Defines the frequency-resolution in combination with
          n_cycles as 'fq_resolution / time_bandwith / T'

    'morlet' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    _check_value(method, METHODS, "method")

    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
        regular_only,
        transfer_only,
    )
    # flatten files
    all_files = list()
    for _, files_ in files.items():
        for _, files__ in files_.items():
            all_files.extend(files__)

    tfr, itc = METHODS[method](all_files, **kwargs)
    return tfr, itc


@fill_doc
def tfr_subject(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    regular_only: bool,
    transfer_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    n_jobs: int = 1,
    **kwargs,
) -> Dict[int, Tuple[AverageTFR, AverageTFR]]:
    """Compute the per-subject TFR.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(regular_only)s
    %(transfer_only)s
    %(participants)s
    method : 'multitaper'
        TFR method used.
    %(n_jobs)s
    **kwargs
        Extra keyword arguments are passed to the TFR method.

    Notes
    -----
    'multitaper' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
        - time_bandwith: Defines the frequency-resolution in combination with
          n_cycles as 'fq_resolution / time_bandwith / T'

    'morlet' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    _check_value(method, METHODS, "method")

    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
        regular_only,
        transfer_only,
    )
    input_pool = [
        (
            list(chain(*values.values())),
            *kwargs.values(),
            participant,
        )
        for participant, values in files.items()
    ]
    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(METHODS[method], input_pool)

    # format as dictionary
    return {idx: (tfr, itc) for tfr, itc, idx in results if tfr is not None}


@fill_doc
def tfr_session(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    n_jobs: int = 1,
    **kwargs,
) -> Dict[int, Dict[int, Tuple[AverageTFR, AverageTFR]]]:
    """Compute the per-session TFR.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(participants)s
    method : 'multitaper'
        TFR method used.
    %(n_jobs)s
    **kwargs
        Extra keyword arguments are passed to the TFR method.

    Notes
    -----
    'multitaper' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
        - time_bandwith: Defines the frequency-resolution in combination with
          n_cycles as 'fq_resolution / time_bandwith / T'

    'morlet' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    _check_value(method, METHODS, "method")

    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
    )
    input_pool = list()
    for participant, files_ in files.items():
        for session, files__ in files_.items():
            input_pool.append(
                (files__, *kwargs.values(), participant, session)
            )

    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(METHODS[method], input_pool)

    # format results
    results_ = dict()
    for tfr, itc, idx, session in results:
        if idx not in results_:
            results_[idx] = dict()
        results_[idx][session] = (tfr, itc)
    return results_


@fill_doc
def tfr_session_groupby(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    n_jobs: int = 1,
    groupby: int = 5,
    **kwargs,
) -> Dict[int, Dict[str, Tuple[AverageTFR, AverageTFR]]]:
    """Compute the per-session TFR. Sessions are group-by blocks.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(participants)s
    method : 'multitaper'
        TFR method used.
    %(n_jobs)s
    groupby : int, 3 or 5
        Number of consecutive session to aggregate together.
    **kwargs
        Extra keyword arguments are passed to the TFR method.

    Notes
    -----
    'multitaper' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
        - time_bandwith: Defines the frequency-resolution in combination with
          n_cycles as 'fq_resolution / time_bandwith / T'

    'morlet' requires:
        - freqs: Frequencies of interest.
        - n_cycles: Defines the time-resolution as 'T = n_cycles / freqs'
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    _check_value(method, METHODS, "method")
    _check_type(groupby, ("int",), "groupby")
    _check_value(groupby, (3, 5), "groupby")

    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
    )
    input_pool = list()
    for participant, files_ in files.items():
        for k, (session, files__) in enumerate(files_.items()):
            if k == 0:
                group_files = list()
                sessions = list()
            elif k % groupby == 0:
                input_pool.append(
                    (group_files, *kwargs.values(), participant, *sessions)
                )
                group_files = list()
                sessions = list()
            group_files.extend(files__)
            sessions.append(session)
        input_pool.append(
            (group_files, *kwargs.values(), participant, *sessions)
        )

    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(METHODS[method], input_pool)

    # format results
    results_ = dict()
    for res in results:
        tfr = res[0]
        itc = res[1]
        idx = res[2]
        sessions = res[3:]
        sessions_str = f"ses-{'-'.join(str(s)for s in sessions)}"
        if idx not in results_:
            results_[idx] = dict()
        results_[idx][sessions_str] = (tfr, itc)
    return results_


# -----------------------------------------------------------------------------
def _load_files(files: List[Path]) -> BaseEpochs:
    """Load the files and concatenate the online runs in an Epochs object."""
    epochs_list = list()
    for file in files:
        raw = read_raw_fif(file, preload=True)
        epochs = make_combine_epochs(raw)
        epochs_list.append(epochs)
        del raw
    try:
        epochs = concatenate_epochs(epochs_list)
    except Exception:
        logger.error("Could not concatenate epochs.")
        return None
    del epochs_list

    return epochs


def _tfr_multitaper(
    files: List[Path],
    freqs: NDArray[float],
    n_cycles: Union[int, NDArray[float]],
    time_bandwidth: int,
    *args,
    **kwargs,
) -> AverageTFR:
    """Compute the TFR representation using multitapers."""
    epochs = _load_files(files)
    if epochs is None:
        return None, None, *args, *kwargs.values()
    tfr, itc = tfr_multitaper(
        epochs, freqs, n_cycles, time_bandwidth, return_itc=True
    )
    return tfr, itc, *args, *kwargs.values()


def _tfr_morlet(
    files: List[Path],
    freqs: NDArray[float],
    n_cycles: Union[int, NDArray[float]],
    *args,
    **kwargs,
) -> AverageTFR:
    """Compute the TFR representation using morlet wavelets."""
    epochs = _load_files(files)
    if epochs is None:
        return None, None, *args, *kwargs.values()
    tfr, itc = tfr_morlet(epochs, freqs, n_cycles, return_itc=True)
    return tfr, itc, *args, *kwargs.values()


METHODS = dict(
    morlet=_tfr_morlet,
    multitaper=_tfr_multitaper,
)
