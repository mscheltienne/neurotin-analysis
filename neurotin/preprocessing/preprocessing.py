from datetime import datetime, timezone
import os
from pathlib import Path
import traceback

import mne

from .bads import PREP_bads_suggestion
from .events import check_events, add_annotations_from_events
from .filters import apply_filter_eeg, apply_filter_aux
from .. import logger
from ..io import read_raw_fif
from ..utils._checks import _check_path, _check_type, _check_value
from ..utils._docs import fill_doc


# -----------------------------------------------------------------------------
@fill_doc
def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, add montage, applying FIR filters, and applying a
    common average reference (CAR).

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    %(bads)s
    """
    # Check sampling frequency
    if raw.info['sfreq'] != 512:
        raw.resample(sfreq=512)

    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)
    raw, _ = add_annotations_from_events(raw)

    # Filter
    apply_filter_aux(raw, bandpass=(1., 40.), notch=True)
    apply_filter_eeg(raw, bandpass=(1., 40.))

    # Mark bad channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy and applies notch
    raw.info['bads'] = bads

    # Add montage
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel

    # CAR
    apply_filter_eeg(raw, car=True)

    return raw, bads


# -----------------------------------------------------------------------------
@fill_doc
def remove_artifact_ic(raw, *, semiauto=False):
    """
    Apply ICA to remove ocular and heartbeat artifacts from raw instance.

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.

    Returns
    -------
    %(raw)s
    ica : ICA
    eog_scores : Scores used for selection of the ocular component(s).
    ecg_scores : Scores used for selection of the heartbeat component(s).
    """
    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')

    # fit ICA
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    ica.fit(raw, picks=picks)

    # select components
    raw_ = raw.copy().pick_types(eeg=True, eog=True, ecg=True, exclude='bads')
    eog_idx, eog_scores = ica.find_bads_eog(
        raw_, threshold=4.8, measure='zscore')
    ecg_idx, ecg_scores = ica.find_bads_ecg(
        raw_, method='correlation', threshold=0.7, measure='correlation')

    # apply ICA
    ica.exclude = eog_idx + ecg_idx

    try:
        assert len(eog_idx) <= 2, 'More than 2 EOG component detected.'
        assert len(ecg_idx) <= 1, 'More than 1 ECG component detected.'
        assert len(ica.exclude) != 0, 'No EOG / ECG component detected.'
    except AssertionError:
        if semiauto:
            ica.plot_scores(eog_scores)
            ica.plot_scores(ecg_scores)
            ica.plot_components(inst=raw_)
            ica.plot_sources(raw_, block=True)
        else:
            raise

    ica.apply(raw)

    # Scores should not be returned when #9846 is fixed.
    return raw, ica, eog_scores, ecg_scores


# -----------------------------------------------------------------------------
@fill_doc
def fill_info(raw):
    """
    Fill the measurement info with:
        - a description including the subject ID, session ID, recording type
        and recording run.
        - device information with the type, model and serial.
        - experimenter name.
        - measurement date (UTC)

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


def _add_description(raw):
    """Add a description including the subject ID, session ID, recording type
    and recording run."""
    fname = Path(raw.filenames[0])
    subject = int(fname.parent.parent.parent.name)
    session = int(fname.parent.parent.name.split()[-1])
    recording_type = fname.parent.name
    recording_run = fname.name.split('-')[0]
    raw.info['description'] = f'Subject {subject} - Session {session} ' + \
                              f'- {recording_type} {recording_run}'


def _add_device_info(raw):
    """Add device information to raw instance."""
    fname = Path(raw.filenames[0])
    raw.info['device_info'] = dict()
    raw.info['device_info']['type'] = 'EEG'
    raw.info['device_info']['model'] = 'eego mylab'
    serial = fname.stem.split('-raw')[0].split('-')[-1].split()[1]
    raw.info['device_info']['serial'] = serial
    raw.info['device_info']['site'] = \
        'https://www.ant-neuro.com/products/eego_mylab'


def _add_experimenter_info(raw, experimenter='Mathieu Scheltienne'):
    """Add experimenter information to raw instance."""
    _check_type(experimenter, (str, ), item_name='experimenter')
    raw.info['experimenter'] = experimenter


def _add_measurement_date(raw):
    """Add measurement date information to raw instance."""
    recording_type_mapping = {
        'Calibration': 'Calib',
        'RestingState': 'RestS',
        'Online': 'OnRun'
        }

    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    _check_value(recording_type, recording_type_mapping, 'recording_type')
    recording_type = recording_type_mapping[recording_type]
    recording_run = int(fname.name.split('-')[0])
    logs_file = fname.parent.parent / 'logs.txt'
    logs_file = _check_path(logs_file, item_name='logs_file', must_exist=True)

    with open(logs_file, 'r') as f:
        lines = f.readlines()
    lines = [line.split(' - ') for line in lines
             if len(line.split(' - ')) > 1]
    logs = [(datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M"),
             line[1].strip(), line[2].strip())
            for line in lines]

    datetime_ = None
    for log in logs:
        if log[1] == recording_type and int(log[2][-1]) == recording_run:
            datetime_ = log[0]
            break
    assert datetime_ is not None

    raw.set_meas_date(datetime_.astimezone(timezone.utc))


def _add_subject_info(raw):
    """Add subject information to raw instance."""
    subject_info = dict()
    # subject ID
    fname = Path(raw.filenames[0])
    subject = int(fname.parent.parent.parent.name)
    subject_info['id'] = subject
    subject_info['his_id'] = str(subject).zfill(3)
    raw.info['subject_info'] = subject_info


# -----------------------------------------------------------------------------
def pipeline(
        fname,
        dir_in,
        dir_out,
        ):
    """Preprocessing pipeline function called on every raw files.

    Add measurement information.

    Parameters
    ----------
    fname : str
        Path to the input file to the processing pipeline.
    dir_in : path-like
        Path to the folder containing the FIF files to process
    dir_out : path-like
        Path to the folder containing the FIF files processed. The FIF files
        are saved under the same relative folder structure as in 'dir_in'.

    Returns
    -------
    success : bool
        False if a processing step raised an Exception.
    fname : str
        Path to the input file to the processing pipeline.
    """
    logger.info('Processing: %s' % fname)
    try:
        # checks paths
        fname = _check_path(fname, item_name='fname', must_exist=True)
        dir_in = _check_path(dir_in, 'dir_in', must_exist=True)
        dir_out = _check_path(dir_out, 'dir_out', must_exist=True)

        # create output file name
        output_fname_raw, output_fname_ica = \
            _create_output_fname(fname, dir_in, dir_out)

        # load
        raw = read_raw_fif(fname)

        # fill info
        raw = fill_info(raw)

        # prepare
        raw, bads = prepare_raw(raw)
        assert len(raw.info['projs']) == 0  # sanity-check

        # ica
        raw, ica, eog_scores, ecg_scores = remove_artifact_ic(raw)

        # interpolate bads
        raw.interpolate_bads(reset_bads=True, mode='accurate')

        # export
        raw.save(output_fname_raw, fmt="double", overwrite=True)
        ica.save(output_fname_ica)

        return (True, str(fname))

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.warning(traceback.format_exc())
        return (False, str(fname))


def _create_output_fname(
        fname,
        dir_in,
        dir_out
        ):
    """Creates the output file names based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(dir_in)
    # create output fname
    output_fname_raw = dir_out / relative_fname
    output_fname_ica = \
        dir_out / str(relative_fname).replace('-raw.fif', '-ica.fif')
    os.makedirs(output_fname_raw.parent, exist_ok=True)
    return output_fname_raw, output_fname_ica
