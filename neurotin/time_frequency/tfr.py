import multiprocessing as mp
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from mne import concatenate_epochs
from mne.io import read_raw_fif
from mne.time_frequency import AverageTFR, tfr_multitaper
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
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    n_jobs: int = 1,
    **kwargs,
) -> AverageTFR:
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
    baseline : None | tuple of float
        Baseline correction applied to the 24 second epochs.
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
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    methods = dict(
        multitaper=_tfr_multitaper,
    )
    _check_value(method, methods, "method")
    if baseline is not None:
        _check_type(baseline, (tuple,), "baseline")
        if len(baseline) != 2:
            raise ValueError("Baseline should be a 2-length tuple.")
        _check_type(baseline[0], ("numeric", None), "baseline_tmin")
        _check_type(baseline[1], ("numeric", None), "baseline_tmax")

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

    tfr = methods[method](all_files, baseline, **kwargs)
    return tfr


@fill_doc
def tfr_subject(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    regular_only: bool,
    transfer_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    n_jobs: int = 1,
    **kwargs,
) -> Dict[int, AverageTFR]:
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
    baseline : None | tuple of float
        Baseline correction applied to the 24 second epochs.
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
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    methods = dict(
        multitaper=_tfr_subject_multitaper,
    )
    _check_value(method, methods, "method")
    if baseline is not None:
        _check_type(baseline, (tuple,), "baseline")
        if len(baseline) != 2:
            raise ValueError("Baseline should be a 2-length tuple.")
        _check_type(baseline[0], ("numeric", None), "baseline_tmin")
        _check_type(baseline[1], ("numeric", None), "baseline_tmax")

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
            participant,
            list(chain(*values.values())),
            baseline,
            *kwargs.values(),
        )
        for participant, values in files.items()
    ]
    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(methods[method], input_pool)

    # format as dictionary
    return {idx: tfr for idx, tfr in results if tfr is not None}


def _tfr_subject_multitaper(
    participant: int,
    files: List[Path],
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    freqs: NDArray[float],
    n_cycles: Union[int, NDArray[float]],
    time_bandwidth: int,
) -> Tuple[int, AverageTFR]:
    """Compute the TFR representation using multitaper at the subject-level."""
    tfr = _tfr_multitaper(files, baseline, freqs, n_cycles, time_bandwidth)
    return participant, tfr


@fill_doc
def tfr_session(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    method: str = "multitaper",
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    n_jobs: int = 1,
    **kwargs,
) -> Dict[int, Dict[int, AverageTFR]]:
    """Compute the per-session TFR.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(participants)s
    method : 'multitaper'
        TFR method used.
    baseline : None | tuple of float
        Baseline correction applied to the 24 second epochs.
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
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    methods = dict(
        multitaper=_tfr_session_multitaper,
    )
    _check_value(method, methods, "method")
    if baseline is not None:
        _check_type(baseline, (tuple,), "baseline")
        if len(baseline) != 2:
            raise ValueError("Baseline should be a 2-length tuple.")
        _check_type(baseline[0], ("numeric", None), "baseline_tmin")
        _check_type(baseline[1], ("numeric", None), "baseline_tmax")

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
                (participant, session, files__, baseline, *kwargs.values())
            )

    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(methods[method], input_pool)

    # format results
    results_ = dict()
    for idx, session, tfr in results:
        if idx not in results_:
            results_[idx] = dict()
        results_[idx][session] = tfr
    return results_


def _tfr_session_multitaper(
    participant: int,
    session: int,
    files: List[Path],
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    freqs: NDArray[float],
    n_cycles: Union[int, NDArray[float]],
    time_bandwidth: int,
):
    """Compute the TFR representaito using multitaper at the session-level."""
    tfr = _tfr_multitaper(files, baseline, freqs, n_cycles, time_bandwidth)
    return participant, session, tfr


def _tfr_multitaper(
    files: List[Path],
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    freqs: NDArray[float],
    n_cycles: Union[int, NDArray[float]],
    time_bandwidth: int,
) -> AverageTFR:
    """Compute the TFR representation of the given files."""
    epochs_list = list()
    for file in files:
        raw = read_raw_fif(file, preload=True)
        epochs = make_combine_epochs(raw)
        if baseline is not None:
            epochs.apply_baseline(baseline)
        epochs_list.append(epochs)
        del raw
    try:
        epochs = concatenate_epochs(epochs_list)
    except Exception:
        logger.error("Could not concatenate epochs.")
        return None
    del epochs_list

    tfr = tfr_multitaper(
        epochs, freqs, n_cycles, time_bandwidth, return_itc=False
    )

    return tfr
