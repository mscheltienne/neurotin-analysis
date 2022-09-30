import multiprocessing as mp
import re
import traceback
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from mne.io import read_raw_fif
from numpy.typing import NDArray
from scipy.integrate import simpson

from .. import logger
from ..utils._checks import (
    _check_n_jobs,
    _check_participants,
    _check_path,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils.selection import list_runs_pp
from .epochs import make_fixed_length_epochs


@fill_doc
def compute_bandpower(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    regular_only: bool,
    transfer_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    duration: float,
    overlap: float,
    fmin: float,
    fmax: float,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the absolute and relative band power of an online recording.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(regular_only)s
    %(transfer_only)s
    %(participants)s
    %(psd_duration)s
    %(psd_overlap)s
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.

    Returns
    -------
    df_bp_abs : DataFrame
        Absolute band power.
    df_bp_rel : DataFrame
        Relative band power.

    Notes
    -----
    The PSD is computing using a multitaper method with adaptive weights.
    The band-power is computed using Simpson's rule for integration.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    _check_type(fmin, ("numeric",), item_name="fmin")
    _check_type(fmax, ("numeric",), item_name="fmax")
    n_jobs = _check_n_jobs(n_jobs)

    # create input_pool
    files = list_runs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
        regular_only,
        transfer_only,
    )
    flatten_files = list()
    for files_dict in files.values():
        for elt in files_dict.values():
            flatten_files.extend(elt)
    input_pool = [
        (file, duration, overlap, fmin, fmax) for file in flatten_files
    ]
    assert 0 < len(input_pool)  # sanity check

    # compute psds
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_compute_bandpower, input_pool)

    # construct dataframe
    bp_abs = dict()
    bp_rel = dict()
    for participant, session, run, psds_abs, psds_rel, ch_names in results:
        for phase in psds_abs:
            _add_data_to_dict(
                bp_abs,
                participant,
                session,
                run,
                phase,
                psds_abs[phase],
                ch_names,
            )
            _add_data_to_dict(
                bp_rel,
                participant,
                session,
                run,
                phase,
                psds_rel[phase],
                ch_names,
            )

    return pd.DataFrame.from_dict(
        bp_abs, orient="columns"
    ), pd.DataFrame.from_dict(bp_rel, orient="columns")


def _compute_bandpower(
    fname: Path,
    duration: float,
    overlap: float,
    fmin: float,
    fmax: float,
) -> Tuple[int, int, int, NDArray[float], NDArray[float], List[str]]:
    """Compute the absolute and relative band power of an online recording."""
    logger.info("Processing: %s" % fname)
    try:
        raw = read_raw_fif(fname, preload=True)
        # find participant id
        participant = int(fname.parent.parent.parent.name)
        # find session id
        pattern = re.compile(r"Session (\d{1,2})")
        session = re.findall(pattern, str(fname))
        assert len(session) == 1
        session = int(session[0])
        # find run id
        run = int(fname.name.split("-")[0])
        # compute psd
        epochs = make_fixed_length_epochs(raw, duration, overlap)
        spectrum = epochs.compute_psd(method="multitaper", adaptive=True)
        # compute band power
        freq_res = spectrum.freqs[1] - spectrum.freqs[0]
        bp_absolute = dict()
        bp_relative = dict()
        for phase in spectrum.event_id:
            psd = spectrum[phase].get_data(fmin=fmin, fmax=fmax)
            psd_full = spectrum[phase].get_data(fmin=1.0, fmax=40.0)
            bp_abs = simpson(psd, dx=freq_res, axis=-1)
            bp_rel = bp_abs / simpson(psd_full, dx=freq_res, axis=-1)
            bp_absolute[phase] = np.average(bp_abs, axis=0)
            bp_relative[phase] = np.average(bp_rel, axis=0)
        # clean up
        del raw

    except Exception:
        logger.warning("FAILED: %s -> Skip." % fname)
        logger.warning(traceback.format_exc())

    return (
        participant,
        session,
        run,
        bp_absolute,
        bp_relative,
        spectrum.ch_names,
    )


def _add_data_to_dict(
    data_dict: dict,
    participant: int,
    session: int,
    run: int,
    phase: str,
    data: NDArray[float],
    ch_names: List[str],
) -> None:
    """Add band-power to data dictionary."""
    keys = ["participant", "session", "run", "phase", "idx"] + ch_names

    # init
    for key in keys:
        if key not in data_dict:
            data_dict[key] = list()

    # fill data
    data_dict["participant"].append(participant)
    data_dict["session"].append(session)
    data_dict["run"].append(run)
    data_dict["phase"].append(phase[:-2])
    data_dict["idx"].append(int(phase[-1]))  # idx of the phase within the run

    # channel psd
    for ch, value in zip(ch_names, data):
        data_dict[ch].append(value)

    # sanity check
    entries = len(data_dict["participant"])
    assert all(len(data_dict[key]) == entries for key in keys)
