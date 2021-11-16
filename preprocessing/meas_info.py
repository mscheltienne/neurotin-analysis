from pathlib import Path
from datetime import datetime, timezone

from utils.checks import _check_type, _check_path, _check_value


RECORDING_TYPE_MAPPING = {
    'Calibration': 'Calib',
    'RestingState': 'RestS',
    'Online': 'OnRun'}


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


def fill_info(raw, subject, sex, birthday):
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
    _add_measurement_date(raw)
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


def _add_measurement_date(raw):
    """Add measurement date information to raw instance."""
    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    _check_value(recording_type, RECORDING_TYPE_MAPPING,
                 item_name='recording_type')
    recording_type = RECORDING_TYPE_MAPPING[recording_type]
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
