import multiprocessing as mp
import re
import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mne import BaseEpochs, pick_types
from mne.io import BaseRaw, read_raw_fif
from mne.time_frequency import EpochsSpectrum
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
from ..utils.list_files import list_raw_fif
from .epochs import make_fixed_length_epochs, reject_epochs


@fill_doc
def psd_avg_band(
    folder,
    participants: Union[int, List[int], Tuple[int, ...]],
    duration: float,
    overlap: float,
    reject: Optional[Union[Dict[str, float], str]],
    fmin: float,
    fmax: float,
    average: str = "mean",
    n_jobs: int = 1,
):
    """Compute the PSD.

    Average by frequency band for the given participants using the welch
    method.

    Parameters
    ----------
    folder : path-like
        Path to the folder containing preprocessed files.
    %(participants)s
    %(psd_duration)s
    %(psd_duration)s
    %(psd_reject)s
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    average : 'mean' | 'integrate'
        How to average the frequency bin/spectrum. Either 'mean' to calculate
        the arithmetic mean of all bins or 'integrate' to use Simpson's rule to
        compute integral from samples.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.

    Returns
    -------
    %(df_psd)s
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    participants = _check_participants(participants)
    _check_type(fmin, ("numeric",), item_name="fmin")
    _check_type(fmax, ("numeric",), item_name="fmax")
    _check_type(average, (str,), item_name="average")
    _check_value(average, ("mean", "integrate"), item_name="average")
    n_jobs = _check_n_jobs(n_jobs)

    # create input_pool
    input_pool = list()
    for participant in participants:
        fnames = list_raw_fif(folder / str(participant).zfill(3))
        for fname in fnames:
            if fname.parent.name != "Online":
                continue
            input_pool.append(
                (
                    participant,
                    fname,
                    duration,
                    overlap,
                    reject,
                    fmin,
                    fmax,
                    average,
                )
            )
    assert 0 < len(input_pool)  # sanity check

    # compute psds
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_psd_avg_band, input_pool)

    # construct dataframe
    psd_dict = dict()
    for participant, session, run, psds, ch_names in results:
        for phase in psds:
            _add_data_to_dict(
                psd_dict,
                participant,
                session,
                run,
                phase,
                psds[phase],
                ch_names,
            )

    return pd.DataFrame.from_dict(psd_dict, orient="columns")


def _psd_avg_band(
    participant,
    fname,
    duration: float,
    overlap: float,
    reject: Optional[Union[Dict[str, float], str]],
    fmin: float,
    fmax: float,
    average: str,
):
    """Compute the PSD.

    Average by frequency band for the given participants using the welch
    method.
    """
    logger.info("Processing: %s" % fname)
    try:
        raw = read_raw_fif(fname, preload=True)
        # find session id
        pattern = re.compile(r"Session (\d{1,2})")
        session = re.findall(pattern, str(fname))
        assert len(session) == 1
        session = int(session[0])
        # find run id
        run = int(fname.name.split("-")[0])
        # compute psds
        psds, freqs = _psd_welch(
            raw, duration, overlap, reject, fmin=fmin, fmax=fmax
        )
        # find channel names
        ch_names = raw.pick_types(eeg=True, exclude=[]).ch_names
        assert len(ch_names) == 64  # sanity check

        psds_ = dict()
        for phase in psds:
            if average == "mean":
                psds_[phase] = np.average(psds[phase], axis=-1)
            elif average == "integrate":
                psds_[phase] = simpson(psds[phase], freqs[phase], axis=-1)
            assert psds_[phase].shape == (64,)  # sanity check

        # clean up
        del raw

    except Exception:
        logger.warning("FAILED: %s -> Skip." % fname)
        logger.warning(traceback.format_exc())

    return participant, session, run, psds_, ch_names


def _psd_welch(
    raw: BaseRaw, duration: float, overlap: float, reject: Optional[Union[Dict[str, float], str]], **kwargs
) -> Dict[str, EpochsSpectrum]:
    """Compute the power spectral density using the welch method."""
    epochs = make_fixed_length_epochs(raw, duration, overlap)
    if reject is not None:
        epochs, _ = reject_epochs(epochs, reject)
    kwargs = _check_kwargs(kwargs, epochs)
    spectrums = dict()
    for phase in epochs.event_id:
        spectrums[phase] = epochs[phase].compute_psd(method="welch", **kwargs)
    return spectrums


def _check_kwargs(kwargs: dict, epochs: BaseEpochs):
    """Check kwargs provided to _compute_psd_welch."""
    if "picks" not in kwargs:
        kwargs["picks"] = pick_types(epochs.info, eeg=True, exclude=[])
    if "n_fft" not in kwargs:
        kwargs["n_fft"] = epochs.times.size
        logger.debug("Argument 'n_fft' set to %i", epochs._data.shape[-1])
    else:
        logger.debug(
            "Argument 'n_fft' was provided and is set to %i", kwargs["n_fft"]
        )
    if "n_overlap" not in kwargs:
        kwargs["n_overlap"] = 0
        logger.debug("Argument 'n_overlap' set to 0")
    else:
        logger.debug(
            "Argument 'n_overlap' was provided and is set to %i",
            kwargs["n_overlap"],
        )
    if "n_per_seg" not in kwargs:
        kwargs["n_per_seg"] = epochs._data.shape[-1]
        logger.debug("Argument 'n_per_seg' set to %i", epochs._data.shape[-1])
    else:
        logger.debug(
            "Argument 'n_per_seg' was provided and is set to %i",
            kwargs["n_per_seg"],
        )
    return kwargs


def _add_data_to_dict(
    data_dict: dict,
    participant: int,
    session: int,
    run: int,
    phase: str,
    data,
    ch_names,
):
    """Add PSD to data dictionary."""
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
    for k in range(data.shape[0]):
        data_dict[f"{ch_names[k]}"].append(data[k])

    # sanity check
    entries = len(data_dict["participant"])
    assert all(len(data_dict[key]) == entries for key in keys)
