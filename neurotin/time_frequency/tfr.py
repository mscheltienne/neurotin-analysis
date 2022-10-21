import multiprocessing as mp
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from mne import concatenate_epochs
from mne.io import read_raw_fif
from mne.time_frequency import AverageTFR, tfr_multitaper
from numpy.typing import NDArray

from ..utils._checks import (
    _check_n_jobs,
    _check_participants,
    _check_path,
    _check_value,
)
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..utils.selection import list_runs_pp
from .epochs import make_combine_epochs


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
    %(n_jobs)s
    **kwargs
        Extra keyword arguments are passed to the TFR method.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    n_jobs = _check_n_jobs(n_jobs)
    methods = dict(
        multitaper=_tfr_subject_multitaper,
    )
    _check_value(method, methods, "method")

    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
        regular_only,
        transfer_only,
    )
    input_pool = [
        (participant, list(chain(*values.values())), *kwargs.values())
        for participant, values in files.items()
    ]
    assert 0 < len(input_pool)  # sanity check
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_tfr_subject_multitaper, input_pool)

    # format as dictionary
    return {idx: tfr for idx, tfr in results if tfr is not None}


def _tfr_subject_multitaper(
    participant: int,
    files: List[Path],
    freqs: NDArray[float] = np.arange(1, 15, 1),
    n_cycles: Union[int, NDArray[float]] = np.arange(1, 15, 1) / 2,
    time_bandwidth: int = 2,
) -> Tuple[int, AverageTFR]:
    """Compute the TFR representation using multitaper at the subject-level."""
    epochs_list = list()
    for file in files:
        raw = read_raw_fif(file, preload=True)
        epochs = make_combine_epochs(raw)
        epochs.apply_baseline((0, 0.5))
        epochs_list.append(epochs)
        del raw
    try:
        epochs = concatenate_epochs(epochs_list)
    except Exception:
        logger.error(
            "Could not concatenate epochs for participant %i", participant
        )
        return participant, None
    del epochs_list

    tfr = tfr_multitaper(
        epochs, freqs, n_cycles, time_bandwidth, return_itc=False
    )

    return participant, tfr
