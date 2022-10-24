import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
from mne import pick_types
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.viz.ica import _prepare_data_ica_properties
from mne_icalabel import label_components

from .. import logger
from ..io import read_raw_fif
from ..utils._checks import _check_path, _check_type, _check_value
from ..utils._docs import fill_doc
from .bads import PREP_bads_suggestion
from .bridge import repair_bridged_electrodes
from .events import (
    add_annotations_from_events,
    check_events,
    find_crop_tmin_tmax,
)
from .filters import apply_filter_aux, apply_filter_eeg


# -----------------------------------------------------------------------------
@fill_doc
def prepare_raw(raw: BaseRaw) -> BaseRaw:
    """Prepare raw object.

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    """
    # drop channels
    raw.drop_channels(["M1", "M2", "PO5", "PO6"])

    # check sampling frequency
    if raw.info["sfreq"] != 512:
        raw.resample(sfreq=512)

    # check events
    recording_type = Path(raw.filenames[0]).stem.split("-")[1]
    check_events(raw, recording_type)
    raw, _ = add_annotations_from_events(raw)

    # fix bridged electrodes
    raw.set_montage("standard_1020")
    raw = repair_bridged_electrodes(raw)
    raw.set_montage(None)

    # filter
    apply_filter_aux(raw, bandpass=(1.0, 40.0), notch=True)
    apply_filter_eeg(raw, bandpass=(1.0, 100.0))

    # mark bad channels, operates on a copy and applies filters
    raw.info["bads"] = PREP_bads_suggestion(raw, prepare_raw=True)

    # add montage
    raw.add_reference_channels(ref_channels="CPz")
    raw.set_montage("standard_1020")  # only after adding ref channel

    # apply CAR
    raw.set_eeg_reference(
        ref_channels="average",
        ch_type="eeg",
        projection=False,
    )

    # crop artifacted beginning
    tmin, tmax = find_crop_tmin_tmax(raw)
    raw.crop(tmin, tmax, include_tmax=True)
    return raw


# -----------------------------------------------------------------------------
@fill_doc
def remove_artifact_ic(raw: BaseRaw) -> BaseRaw:
    """Apply ICA to remove artifact-related independent components.

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    %(ica)s
    """
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    ica = ICA(
        n_components=None,  # should be picks.size - 1
        method="picard",
        max_iter="auto",
        fit_params=dict(ortho=False, extended=True),
    )
    ica.fit(raw, picks=picks)

    # run iclabel
    component_dict = label_components(raw, ica, method="iclabel")

    # keep only brain components
    labels = component_dict["labels"]
    exclude = [k for k, name in enumerate(labels) if name != "brain"]

    # compute variances on epochs
    kind, dropped_indices, epochs_src, data = _prepare_data_ica_properties(
        raw, ica, reject_by_annotation=True, reject="auto")
    ica_data = np.swapaxes(data, 0, 1)
    var = np.var(ica_data, axis=2)  # (n_components, n_epochs)
    # TODO: Use var to identify which ICs don't need to be dropped because they
    # impact the overall signal only punctualy.
    # Ocular and Cardiac related components should be removed anyway,
    # regardless of the variance.

    # apply ICA
    ica.exclude = exclude
    ica.apply(raw)

    return raw, ica


# -----------------------------------------------------------------------------
@fill_doc
def fill_info(raw: BaseRaw) -> BaseRaw:
    """Fill the measurement info.

    The filled entries are:
        - a description including the subject ID, session ID, recording type
        and recording run.
        - device information with the type, model and serial.
        - experimenter name.
        - measurement date (UTC).

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    """
    functions = (
        _add_device_info,
        _add_experimenter_info,
        _add_subject_info,
        _add_measurement_date,
    )
    for function in functions:
        try:
            function(raw)
        except Exception:
            pass
    raw.info._check_consistency()
    return raw


def _add_description(raw: BaseRaw) -> None:
    """Add a description.

    The description includes the subject ID, session ID, recording type
    and recording run.
    """
    fname = Path(raw.filenames[0])
    subject = int(fname.parent.parent.parent.name)
    session = int(fname.parent.parent.name.split()[-1])
    recording_type = fname.parent.name
    recording_run = fname.name.split("-")[0]
    raw.info["description"] = (
        f"Subject {subject} - Session {session} "
        + f"- {recording_type} {recording_run}"
    )


def _add_device_info(raw: BaseRaw) -> None:
    """Add device information to raw instance."""
    fname = Path(raw.filenames[0])
    raw.info["device_info"] = dict()
    raw.info["device_info"]["type"] = "EEG"
    raw.info["device_info"]["model"] = "eego mylab"
    serial = fname.stem.split("-raw")[0].split("-")[-1].split()[1]
    raw.info["device_info"]["serial"] = serial
    raw.info["device_info"][
        "site"
    ] = "https://www.ant-neuro.com/products/eego_mylab"


def _add_experimenter_info(
    raw: BaseRaw, experimenter: str = "Mathieu Scheltienne"
) -> None:
    """Add experimenter information to raw instance."""
    _check_type(experimenter, (str,), item_name="experimenter")
    raw.info["experimenter"] = experimenter


def _add_measurement_date(raw: BaseRaw) -> None:
    """Add measurement date information to raw instance."""
    recording_type_mapping = {
        "Calibration": "Calib",
        "RestingState": "RestS",
        "Online": "OnRun",
    }

    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    _check_value(recording_type, recording_type_mapping, "recording_type")
    recording_type = recording_type_mapping[recording_type]
    recording_run = int(fname.name.split("-")[0])
    logs_file = fname.parent.parent / "logs.txt"
    logs_file = _check_path(logs_file, item_name="logs_file", must_exist=True)

    with open(logs_file, "r") as f:
        lines = f.readlines()
    lines = [line.split(" - ") for line in lines if len(line.split(" - ")) > 1]
    logs = [
        (
            datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M"),
            line[1].strip(),
            line[2].strip(),
        )
        for line in lines
    ]

    datetime_ = None
    for log in logs:
        if log[1] == recording_type and int(log[2][-1]) == recording_run:
            datetime_ = log[0]
            break
    assert datetime_ is not None

    raw.set_meas_date(datetime_.astimezone(timezone.utc))


def _add_subject_info(raw: BaseRaw) -> None:
    """Add subject information to raw instance."""
    subject_info = dict()
    # subject ID
    fname = Path(raw.filenames[0])
    subject = int(fname.parent.parent.parent.name)
    subject_info["id"] = subject
    subject_info["his_id"] = str(subject).zfill(3)
    raw.info["subject_info"] = subject_info


# -----------------------------------------------------------------------------
@fill_doc
def preprocess(fname) -> Tuple[BaseRaw, BaseRaw, ICA]:
    """Preprocess a raw .fif file.

    Parameters
    ----------
    fname : path-like
        Path to the input file to the processing pipeline.

    Returns
    -------
    %(raw)s
    raw_pre_ica : Raw
        MNE raw object used to fit the ICA.
    %(ica)s
    """
    # load
    raw = read_raw_fif(fname)
    # fill info
    raw = fill_info(raw)
    # prepare
    raw = prepare_raw(raw)
    assert len(raw.info["projs"]) == 0  # sanity-check
    # ica
    raw_pre_ica = raw.copy()
    raw, ica = remove_artifact_ic(raw)
    # refilter
    apply_filter_eeg(raw, bandpass=(1.0, 40.0))
    # interpolate bads
    bads = PREP_bads_suggestion(raw, prepare_raw=False)  # run again
    bads = set(bads).union(set(raw.info["bads"]))
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, mode="accurate")

    return raw, raw_pre_ica, ica


# -----------------------------------------------------------------------------
def pipeline(
    fname,
    dir_in,
    dir_out,
) -> Tuple[bool, str]:
    """Preprocessing pipeline function called on every raw files.

    Parameters
    ----------
    fname : path-like
        Path to the file inputted into the processing pipeline.
    dir_in : path-like
        Path to the folder containing the FIF files to process.
    dir_out : path-like
        Path to the folder containing the FIF files processed. The FIF files
        are saved under the same relative folder structure as in 'dir_in'.

    Returns
    -------
    success : bool
        False if a processing step raised an Exception.
    fname : str
        Path to the file inputted into the processing function.
    """
    logger.info("Processing: %s", fname)
    try:
        # checks paths
        fname = _check_path(fname, item_name="fname", must_exist=True)
        dir_in = _check_path(dir_in, "dir_in", must_exist=True)
        dir_out = _check_path(dir_out, "dir_out", must_exist=True)

        # create output file name
        (
            output_fname_raw,
            output_fname_raw_pre_ica,
            output_fname_ica,
        ) = _create_output_fname(fname, dir_in, dir_out)

        # preprocess
        raw, raw_pre_ica, ica = preprocess(fname)

        # export
        raw.save(output_fname_raw, fmt="double", overwrite=True)
        raw_pre_ica.save(
            output_fname_raw_pre_ica, fmt="double", overwrite=True
        )
        ica.save(output_fname_ica)
        return (True, str(fname))

    except Exception:
        logger.warning("FAILED: %s -> Skip.", fname)
        logger.warning(traceback.format_exc())
        return (False, str(fname))


def _create_output_fname(
    fname: Path, dir_in: Path, dir_out: Path
) -> Tuple[Path, Path, Path]:
    """Create the output file names.

    The output file names is based on the relative path between 'fname'
    and 'dir_in'.
    """
    # this will fail if fname is not in dir_in
    relative_fname = fname.relative_to(dir_in)
    # create output fname
    output_fname_raw = dir_out / relative_fname
    output_fname_raw_pre_ica = dir_out / str(relative_fname).replace(
        "-raw", "-pre-ica-raw"
    )
    output_fname_ica = dir_out / str(relative_fname).replace(
        "-raw.fif", "-ica.fif"
    )
    os.makedirs(output_fname_raw.parent, exist_ok=True)
    return output_fname_raw, output_fname_raw_pre_ica, output_fname_ica
