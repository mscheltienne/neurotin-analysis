import os
import re
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp
from datetime import datetime, timezone

import mne

from utils.list_files import raw_fif_selection
from utils.checks import _check_type, _check_path, _check_value, _check_n_jobs


def parse_subject_info(fname):
    """
    Parse the subject_info file and return the subject ID and sex.

    Parameters
    ----------
    fname : str | Path
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
    fname = _check_path(fname, item_name='fname', must_exist=True)
    with open(fname, 'r') as file:
        lines = file.readlines()
    lines = [line.strip().split(';') for line in lines if len(line) > 0]
    lines = [[eval(l.strip()) for l in line]
             for line in lines if len(line) == 3]
    return {line[0]: (line[1], line[2]) for line in lines}


def fill_info(raw, input_dir_fif, raw_dir_fif, subject, sex, birthday):
    """
    Fill the measurement info with:
        - a description including the subject ID, session ID, recording type
        and recording run.
        - device information with the type, model and serial.
        - experimenter name.
        - measurement date (UTC)

    Parameters
    ----------
    raw : Raw
        Raw instance modified in-place.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname) (used to set
        measurement date).
    raw_dir_fif : str | Path
        Path to the directory containing raw data with logs files (used to set
        measurement date).
    subject : int
        ID of the subject.
    sex : int
        Sex of the subject. 1: Male - 2: Female.
    birthday : 3-length tuple of int
        Subject's birthday as (year, month, day).

    Returns
    -------
    raw : Raw instance modified in-place.
    """
    _add_description(raw, subject)
    _add_device_info(raw)
    _add_experimenter_info(raw, experimenter='Mathieu Scheltienne')
    _add_measurement_date(raw, input_dir_fif, raw_dir_fif)
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
    raw.info['description'] = f'Subject {subject} - Session {session} '+\
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


def _add_measurement_date(raw, input_dir_fif, raw_dir_fif):
    """Add measurement date information to raw instance."""
    recording_type_mapping = {
        'Calibration': 'Calib',
        'RestingState': 'RestS',
        'Online': 'OnRun'}

    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    _check_value(recording_type, recording_type_mapping,
                 item_name='recording_type')
    recording_type = recording_type_mapping[recording_type]
    recording_run = int(fname.name.split('-')[0])
    relative_path = fname.relative_to(input_dir_fif)

    logs_file = raw_dir_fif / relative_path.parent.parent / 'logs.txt'
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
        assert 1900 <= birthday[0] <= 2020
        assert 1 <= birthday[1] <= 12
        assert 1 <= birthday[2] <= 31
    except Exception:
        birthday = None
    return birthday


def pipeline(fname, input_dir_fif, output_dir_fif, raw_dir_fif, subject, sex,
             birthday):
    """
    Pipeline function called on each raw file.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname).
    output_dir_fif : str | Path | None
        Path used to save raw in MNE format with the same structure as in
        fname. If None, the input file is overwritten in-place.
    raw_dir_fif : str | Path
        Path to the directory containing raw data with logs files (used to set
        measurement date).
    subject : int
        ID of the subject.
    sex : int
        Sex of the subject. 1: Male - 2: Female.
    birthday : 3-length tuple of int
        Subject's birthday as (year, month, day).

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to preprocess.
    """
    print ('Preprocessing: %s' % fname)
    try:
        # checks paths
        fname = _check_path(fname, item_name='fname', must_exist=True)
        input_dir_fif = _check_path(input_dir_fif,
                                    item_name='input_dir_fif',
                                    must_exist=True)
        if output_dir_fif is not None:
            output_dir_fif = _check_path(output_dir_fif,
                                         item_name='output_dir_fif',
                                         must_exist=True)
        raw_dir_fif = _check_path(raw_dir_fif, item_name='raw_dir_fif',
                                  must_exist=True)

        # create output file name
        if output_dir_fif is not None:
            output_fname = _create_output_fname(fname, input_dir_fif,
                                                output_dir_fif)
        else:
            output_fname = fname

        raw = mne.io.read_raw_fif(fname, preload=True)
        raw = fill_info(raw, input_dir_fif, raw_dir_fif, subject, sex,
                        birthday)
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname))

    except Exception:
        print ('----------------------------------------------')
        print ('FAILED: %s -> Skip.' % fname)
        print(traceback.format_exc())
        print ('----------------------------------------------')
        return (False, str(fname))


def _create_output_fname(fname, input_dir_fif, output_dir_fif):
    """Creates the output file name based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(input_dir_fif)
    # create output fname
    output_fname = output_dir_fif / relative_fname
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname


def main(input_dir_fif, output_dir_fif, raw_dir_fif, subject_info, n_jobs=1,
         subject=None, session=None, fname=None, ignore_existing=True):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files to preprocess.
    output_dir_fif : str | Path | None
        Path to the folder containing the FIF files preprocessed. If None, the
        input file is overwritten in-place.
    raw_dir_fif : str | Path
        Path to the directory containing raw data with logs files (used to set
        measurement date).
    subject_info : str | Path
        Path to the subject_info file.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).
    ignore_existing : bool
        If True, files already preprocessed are not included.
        If output_dir_fif is None, this is override to False.
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, item_name='input_dir_fif',
                                must_exist=True)
    if output_dir_fif is not None:
        output_dir_fif = _check_path(output_dir_fif,
                                     item_name='output_dir_fif')
        os.makedirs(output_dir_fif, exist_ok=True)
    else:
        output_dir_fif = input_dir_fif
        ignore_existing = False  # overwrite existing files
    subject_info = _check_path(subject_info, item_name='subject_info',
                               must_exist=True)
    n_jobs = _check_n_jobs(n_jobs)

    # read subject_info
    subject_info_dict = parse_subject_info(subject_info)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude=[],
                                subject=subject, session=session, fname=fname,
                                ignore_existing=ignore_existing)

    # create input pool for pipeline based on provided subject info
    input_pool = _create_input_pool(fifs_in, input_dir_fif, output_dir_fif,
                                    raw_dir_fif, subject_info_dict)
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_fif/'fails.pcl', mode='wb') as f:
        pickle.dump([file for success, file in results if not success], f, -1)


def _create_input_pool(fifs_in, input_dir_fif, output_dir_fif, raw_dir_fif,
                       subject_info_dict):
    """Create input pool for pipeline function.
    Shape: (fname, input_dir_fif, output_dir_fif, subject, sex, birthday)."""
    input_pool = list()
    pattern = re.compile('\/(\d{3})\/')  # pattern to match subject
    for fname in fifs_in:
        match = re.findall(pattern, str(fname))
        assert len(match) == 1
        subject = int(match[0])
        sex, birthday = subject_info_dict[subject]
        input_pool.append(
            (fname, input_dir_fif, output_dir_fif, raw_dir_fif,
             subject, sex, birthday))
    return input_pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin preprocessing pipeline to fill measurement '
             'information.',
        description='Fill measurement information for NeuroTin raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_fif', type=str,
        help='folder containing FIF files preprocessed (can be None to '
             'overwrite existing files in input_dir_fif).')
    parser.add_argument(
        'raw_dir_fif', type=str,
        help='folder containing raw FIF files with session logs.')
    parser.add_argument(
        'subject_info', type=str,
        help='path to the subject info file.')
    parser.add_argument(
        '--n_jobs', type=int, metavar='int',
        help='number of parallel jobs.', default=1)
    parser.add_argument(
        '--subject', type=int, metavar='int',
        help='restrict to files with this subject ID.', default=None)
    parser.add_argument(
        '--session', type=int, metavar='int',
        help='restrict with files with this session ID.', default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='restrict to this file.', default=None)
    parser.add_argument(
        '--ignore_existing', action='store_true',
        help='ignore files already processed (set to False if output_dir_fif '
             'is set to None).')

    args = parser.parse_args()

    output_dir_fif = None if args.output_dir_fif.lower().strip() == 'none' \
        else args.output_dir_fif

    main(args.input_dir_fif, output_dir_fif, args.raw_dir_fif,
         args.subject_info, args.n_jobs, args.subject, args.session,
         args.fname, args.ignore_existing)
