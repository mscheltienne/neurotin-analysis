from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import re
import traceback

import mne

from .. import logger
from ..io.cli import write_results
from ..utils.list_files import raw_fif_selection
from ..utils._docs import fill_doc
from ..utils._checks import (_check_type, _check_path, _check_value,
                             _check_n_jobs)


def parse_subject_info(fname_subject_info):
    """
    Parse the subject info file and return the subject ID, sex and birthday.

    Parameters
    ----------
    fname_subject_info : path-like
        Path to the subject info file.

    Returns
    -------
    dict
        key : int
            ID of the subject.
        value : tuple (sex, birthday)
            sex : int
                Sex of the subject. 1: Male - 2: Female.
            birthday : tuple
                3-length tuple (year, month, day)
    """
    fname = _check_path(fname_subject_info, item_name='fname_subject_info',
                        must_exist=True)
    with open(fname, 'r') as file:
        lines = file.readlines()
    lines = [line.strip().split(';') for line in lines if len(line) > 0]
    lines = [[eval(line_.strip()) for line_ in line]
             for line in lines if len(line) == 3]
    return {line[0]: (line[1], line[2]) for line in lines}


@fill_doc
def fill_info(
        raw,
        dir_in,  # contains preprocessed .fif files on which info are filled
        folder,  # contains all raw data (with original date)
        subject,
        sex,
        birthday):
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
    %(dir_in)s
    %(folder_data)s
    %(subject)s
    %(sex)s
    %(birthday)s

    Returns
    -------
    %(raw)s
    """
    _add_description(raw, subject)
    _add_device_info(raw)
    _add_experimenter_info(raw, experimenter='Mathieu Scheltienne')
    _add_measurement_date(raw, dir_in, folder)
    _add_subject_info(raw, subject, sex, birthday)
    raw.info._check_consistency()
    return raw


def _add_description(raw, subject):
    """Add a description including the subject ID, session ID, recording type
    and recording run."""
    fname = Path(raw.filenames[0])
    assert int(fname.parent.parent.parent.name) == subject
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


def _add_measurement_date(raw, dir_in, folder):
    """Add measurement date information to raw instance."""
    recording_type_mapping = {
        'Calibration': 'Calib',
        'RestingState': 'RestS',
        'Online': 'OnRun'
        }

    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    _check_value(recording_type, recording_type_mapping,
                 item_name='recording_type')
    recording_type = recording_type_mapping[recording_type]
    recording_run = int(fname.name.split('-')[0])
    relative_path = fname.relative_to(dir_in)

    logs_file = folder / relative_path.parent.parent / 'logs.txt'
    logs_file = _check_path(logs_file, item_name='logs_file')
    if not logs_file.exists():
        return  # don't set meas date

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


def _add_subject_info(raw, subject, sex, birthday):
    """Add subject information to raw instance."""
    subject_info = dict()
    # subject ID
    subject = _check_subject(subject, raw)
    if subject is not None:
        subject_info['id'] = subject
        subject_info['his_id'] = str(subject).zfill(3)
    # subject sex - (0, 1, 2) for (Unknown, Male, Female)
    subject_info['sex'] = _check_sex(sex)
    # birthday
    subject_info['birthday'] = _check_birthday(birthday)

    # use (future) setter
    raw.info['subject_info'] = subject_info


def _check_subject(subject, raw):
    """Checks that the subject ID is valid."""
    try:
        subject = int(subject)
        fname = Path(raw.filenames[0])
        assert int(fname.parent.parent.parent.name) == subject
    except Exception:
        subject = None
    return subject


def _check_sex(sex):
    """Checks that sex is either 1 for Male or 2 for Female. Else returns 0 for
    unknown."""
    _check_type(sex, ('int', ), item_name='sex')
    sex = sex if sex in (1, 2) else 0
    return sex


def _check_birthday(birthday):
    """Checks that birthday is given as a tuple of int (year, month, day)."""
    try:
        birthday = tuple([int(n) for n in birthday])
        assert 1920 <= birthday[0] <= 2020
        assert 1 <= birthday[1] <= 12
        assert 1 <= birthday[2] <= 31
    except Exception:
        birthday = None
    return birthday


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(
        fname,
        dir_in,
        dir_out,
        folder,
        subject,
        sex,
        birthday
        ):
    """%(pipeline_header)s

    Add measurement information.

    Parameters
    ----------
    %(fname)s
    %(dir_in)s
    %(dir_out)s If None, 'dir_in' is used as 'dir_out'.
    %(folder_data)s
    %(subject)s
    %(sex)s
    %(birthday)s

    Returns
    -------
    %(success)s
    %(fname)s
    """
    logger.info('Processing: %s' % fname)
    try:
        # checks paths
        fname = _check_path(fname, item_name='fname', must_exist=True)
        dir_in = _check_path(dir_in, 'dir_in', must_exist=True)
        if dir_out is not None:
            dir_out = _check_path(dir_out, 'dir_out', must_exist=True)
        folder = _check_path(folder, 'folder', must_exist=True)

        # create output file name
        if dir_out is not None:
            output_fname = _create_output_fname(fname, dir_in, dir_out)
        else:
            output_fname = fname

        raw = mne.io.read_raw_fif(fname, preload=True)
        raw = fill_info(raw, dir_in, folder, subject, sex, birthday)
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname))

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname))


def _create_output_fname(
        fname,
        dir_in,
        dir_out
        ):
    """Creates the output file name based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(dir_in)
    # create output fname
    output_fname = dir_out / relative_fname
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname


@fill_doc
def _cli(
        dir_in,
        dir_out,
        folder,
        fname_subject_info,
        n_jobs=1,
        participant=None,
        session=None,
        fname=None,
        ignore_existing=True
        ):
    """%(cli_header)s

    Parameters
    ----------
    %(dir_in)s
    %(dir_out)s If None, 'dir_in' is used as 'dir_out'.
    %(folder_data)s
    fname_subject_info : path-like
        Path to the subject info file.
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    %(ignore_existing)s
    """
    # check arguments
    dir_in = _check_path(dir_in, 'dir_in', must_exist=True)
    if dir_out is not None:
        dir_out = _check_path(dir_out, 'dir_out')
        os.makedirs(dir_out, exist_ok=True)
    else:
        dir_out = dir_in
        ignore_existing = False  # overwrite existing files
    folder = _check_path(folder, 'folder', must_exist=True)
    fname_subject_info = _check_path(fname_subject_info, 'fname_subject_info',
                                     must_exist=True)
    n_jobs = _check_n_jobs(n_jobs)

    # read subject_info
    subject_info_dict = parse_subject_info(fname_subject_info)

    # list files to process
    fifs = raw_fif_selection(
        dir_in,
        dir_out,
        participant=participant,
        session=session,
        fname=fname,
        ignore_existing=ignore_existing
        )

    # create input pool for pipeline based on provided subject info
    input_pool = _create_input_pool(
        fifs,
        dir_in,
        dir_out,
        folder,
        subject_info_dict
        )
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    write_results(results, dir_out / 'meas_info.pcl')


def _create_input_pool(
        fifs,
        dir_in,
        dir_out,
        folder,
        subject_info_dict
        ):
    """Create input pool for pipeline function.
    Shape: (fname, dir_in, dir_out, folder, subject, sex, birthday)."""
    input_pool = list()
    # pattern to match subject
    pattern = re.compile(r'\%s(\d{3})\%s' % (os.sep, os.sep))
    for fname in fifs:
        match = re.findall(pattern, str(fname))
        assert len(match) == 1
        subject = int(match[0])
        try:
            sex, birthday = subject_info_dict[subject]
        except KeyError:
            sex, birthday = 0, None
        input_pool.append(
            (fname, dir_in, dir_out, folder, subject, sex, birthday))
    return input_pool
