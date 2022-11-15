import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mne import pick_types
from mne.io import read_raw_fif
from numpy.typing import NDArray
from scipy.integrate import simpson

from .. import logger
from ..utils._checks import (
    _check_n_jobs,
    _check_participants,
    _check_path,
    _check_type,
    _check_value,
)
from ..utils._docs import fill_doc
from ..utils.selection import list_rs_pp, list_runs_pp
from .epochs import make_epochs


@fill_doc
def compute_bandpower_onrun(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    regular_only: bool,
    transfer_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    fmin: float,
    fmax: float,
    duration: float = 2,
    overlap: float = 1.9,
    folder_weights: Optional[Union[str, Path]] = None,
    weights: Optional[str] = None,
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
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    duration : float
        Duration of a welch's segment in seconds.
    overlap : float
        Overlap between 2 welch's segment in seconds.
    folder_weights : path-like | None
        Folder containing the subject and group level electrode weights.
    weights : str | None
        Either "group" or "subject" for group or subject level weights.
    %(n_jobs)s

    Returns
    -------
    df_bp_abs : DataFrame
        Absolute band power.
    df_bp_rel : DataFrame
        Relative band power.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    _check_type(fmin, ("numeric",), item_name="fmin")
    _check_type(fmax, ("numeric",), item_name="fmax")
    assert 0 < fmin
    assert 0 < fmax
    _check_type(duration, ("numeric",), item_name="duration")
    _check_type(overlap, ("numeric",), item_name="overlap")
    assert 0 < duration
    assert 0 < overlap
    assert overlap < duration
    _check_type(weights, (str, None), "weights")
    if folder_weights is not None:
        folder_weights = _check_path(
            folder_weights, "folder_weights", must_exist=True
        )
        assert weights is not None
        _check_value(weights, ("group", "subject"), "weights")
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
        (file, fmin, fmax, duration, overlap, folder_weights, weights)
        for file in flatten_files
    ]
    assert 0 < len(input_pool)  # sanity check

    # compute psds
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_compute_bandpower_onrun, input_pool)

    # construct dataframe
    bp_abs = dict()
    bp_rel = dict()
    for participant, session, run, psds_abs, psds_rel, ch_names in results:
        _add_data_to_dict_onrun(
            bp_abs,
            participant,
            session,
            run,
            psds_abs,
            ch_names,
        )
        _add_data_to_dict_onrun(
            bp_rel,
            participant,
            session,
            run,
            psds_rel,
            ch_names,
        )

    # convert to dataframes
    bp_abs = pd.DataFrame.from_dict(bp_abs, orient="columns")
    bp_rel = pd.DataFrame.from_dict(bp_rel, orient="columns")

    return bp_abs, bp_rel


def _compute_bandpower_onrun(
    fname: Path,
    fmin: float,
    fmax: float,
    duration: float,
    overlap: float,
    folder_weights: Optional[Path],
    weights: Optional[str],
) -> Tuple[int, int, int, NDArray[float], NDArray[float], List[str]]:
    """Compute the absolute and relative band power of an online recording."""
    logger.info("Processing: %s", fname)
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

    # convert durations to samples
    n_fft = int(duration * raw.info["sfreq"])
    n_overlap = int(overlap * raw.info["sfreq"])

    # load and apply weights if needed
    if weights is not None:
        if weights == "group":
            weights = "avg.pcl"
        elif weights == "subject":
            weights = f"{str(participant).zfill(3)}.pcl"
        weights = pd.read_pickle(folder_weights / weights)

        picks = [raw.ch_names[k] for k in pick_types(raw.info, eeg=True)]
        weights = [weights[ch] for ch in picks]
        raw.apply_function(
            lambda x: (x.T * weights).T, picks=picks, channel_wise=False
        )
    # note: weights are completely useless now that the rest and regulation
    # phase are divided for baseline correction. The weight factor cancels
    # itself out in the division.

    # create regulation / non-regulation epochs
    epochs = make_epochs(raw)
    # clean up
    del raw
    # compute PSD
    spectrum_reg = epochs["regulation"].compute_psd(
        method="welch",
        n_fft=n_fft,
        n_overlap=n_overlap,
        tmin=1.0,
        tmax=epochs["regulation"].times[-1] - 1,
        fmin=1.0,
        fmax=40.0,
    )
    spectrum_rest = epochs["non-regulation"].compute_psd(
        method="welch",
        n_fft=n_fft,
        n_overlap=n_overlap,
        tmin=1.0,
        tmax=epochs["non-regulation"].times[-1] - 1,
        fmin=1.0,
        fmax=40.0,
    )
    assert np.allclose(spectrum_reg.freqs, spectrum_rest.freqs)
    assert spectrum_reg.ch_names == spectrum_rest.ch_names

    # compute the band-power
    freq_res = spectrum_reg.freqs[1] - spectrum_reg.freqs[0]
    psd = spectrum_reg.get_data(fmin=fmin, fmax=fmax)
    psd_full = spectrum_reg.get_data(fmin=1.0, fmax=40.0)
    bp_abs_reg = simpson(psd, dx=freq_res, axis=-1)
    bp_rel_reg = bp_abs_reg / simpson(psd_full, dx=freq_res, axis=-1)
    del psd_full
    del psd
    psd = spectrum_rest.get_data(fmin=fmin, fmax=fmax)
    psd_full = spectrum_rest.get_data(fmin=1.0, fmax=40.0)
    bp_abs_rest = simpson(psd, dx=freq_res, axis=-1)
    bp_rel_rest = bp_abs_rest / simpson(psd_full, dx=freq_res, axis=-1)
    del psd_full
    del psd

    # baseline-correction by dividing both
    bp_abs = bp_abs_reg / bp_abs_rest
    bp_rel = bp_rel_reg / bp_rel_rest

    # clean up
    del bp_abs_reg
    del bp_abs_rest
    del bp_rel_reg
    del bp_rel_rest
    del epochs

    return (
        participant,
        session,
        run,
        bp_abs,
        bp_rel,
        spectrum_reg.ch_names,
    )


def _add_data_to_dict_onrun(
    data_dict: Dict[str, Union[float, str]],
    participant: int,
    session: int,
    run: int,
    data: NDArray[float],
    ch_names: List[str],
) -> None:
    """Add band-power to data dictionary."""
    keys = ["participant", "session", "run", "idx"] + ch_names

    # init
    for key in keys:
        if key not in data_dict:
            data_dict[key] = list()

    # fill data
    for k, epoch in enumerate(data):
        data_dict["participant"].append(participant)
        data_dict["session"].append(session)
        data_dict["run"].append(run)
        data_dict["idx"].append(k + 1)  # idx of the phase within the run
        # channel psd
        for ch, value in zip(ch_names, epoch):
            data_dict[ch].append(value)

    # sanity check
    entries = len(data_dict["participant"])
    assert all(len(data_dict[key]) == entries for key in keys)


@fill_doc
def compute_bandpower_rs(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    valid_only: bool,
    participants: Union[int, List[int], Tuple[int, ...]],
    fmin: float,
    fmax: float,
    duration: float = 2,
    overlap: float = 1.9,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the absolute and relative band power of a resting-state.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(valid_only)s
    %(participants)s
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    duration : float
        Duration of a welch's segment in seconds.
    overlap : float
        Overlap between 2 welch's segment in seconds.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.

    Returns
    -------
    df_bp_abs : DataFrame
        Absolute band power.
    df_bp_rel : DataFrame
        Relative band power.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    folder_pp = _check_path(folder_pp, item_name="folder_pp", must_exist=True)
    participants = _check_participants(participants)
    _check_type(fmin, ("numeric",), item_name="fmin")
    _check_type(fmax, ("numeric",), item_name="fmax")
    assert 0 < fmin
    assert 0 < fmax
    _check_type(duration, ("numeric",), item_name="duration")
    _check_type(overlap, ("numeric",), item_name="overlap")
    assert 0 < duration
    assert 0 < overlap
    assert overlap < duration
    n_jobs = _check_n_jobs(n_jobs)

    files = list_rs_pp(
        folder,
        folder_pp,
        participants,
        valid_only,
    )
    flatten_files = list()
    for files_dict in files.values():
        for elt in files_dict.values():
            flatten_files.extend(elt)
    input_pool = [
        (file, fmin, fmax, duration, overlap) for file in flatten_files
    ]
    assert 0 < len(input_pool)  # sanity check

    # compute psds
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_compute_bandpower_rs, input_pool)

    # construct dataframe
    bp_abs = dict()
    bp_rel = dict()
    for participant, session, psds_abs, psds_rel, ch_names in results:
        _add_data_to_dict_rs(
            bp_abs,
            participant,
            session,
            psds_abs,
            ch_names,
        )
        _add_data_to_dict_rs(
            bp_rel,
            participant,
            session,
            psds_rel,
            ch_names,
        )

    return pd.DataFrame.from_dict(
        bp_abs, orient="columns"
    ), pd.DataFrame.from_dict(bp_rel, orient="columns")


def _compute_bandpower_rs(
    fname: Path,
    fmin: float,
    fmax: float,
    duration: float,
    overlap: float,
) -> Tuple[int, int, NDArray[float], NDArray[float], List[str]]:
    """Compute the absolute and relative band power of a resting-state."""
    logger.info("Processing: %s", fname)
    raw = read_raw_fif(fname, preload=True)
    # find participant id
    participant = int(fname.parent.parent.parent.name)
    # find session id
    pattern = re.compile(r"Session (\d{1,2})")
    session = re.findall(pattern, str(fname))
    assert len(session) == 1
    session = int(session[0])
    # convert durations to samples
    n_fft = int(duration * raw.info["sfreq"])
    n_overlap = int(overlap * raw.info["sfreq"])
    # compute psd
    spectrum = raw.compute_psd(
        method="welch",
        n_fft=n_fft,
        n_overlap=n_overlap,
        tmin=1,
        tmax=raw.times[-1] - 1,
        fmin=1.0,
        fmax=40.0,
    )
    # compute band power
    freq_res = spectrum.freqs[1] - spectrum.freqs[0]
    psd = spectrum.get_data(fmin=fmin, fmax=fmax)
    psd_full = spectrum.get_data(fmin=1.0, fmax=40.0)
    bp_abs = simpson(psd, dx=freq_res, axis=-1)
    bp_rel = bp_abs / simpson(psd_full, dx=freq_res, axis=-1)
    # clean up
    del raw

    return (participant, session, bp_abs, bp_rel, spectrum.ch_names)


def _add_data_to_dict_rs(
    data_dict: Dict[str, float],
    participant: int,
    session: int,
    data: NDArray[float],
    ch_names: List[str],
) -> None:
    """Add band-power to data dictionary."""
    keys = ["participant", "session"] + ch_names

    # init
    for key in keys:
        if key not in data_dict:
            data_dict[key] = list()

    # fill data
    data_dict["participant"].append(participant)
    data_dict["session"].append(session)

    # channel psd
    for ch, value in zip(ch_names, data):
        data_dict[ch].append(value)

    # sanity check
    entries = len(data_dict["participant"])
    assert all(len(data_dict[key]) == entries for key in keys)
