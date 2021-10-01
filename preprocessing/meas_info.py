from pathlib import Path
from datetime import datetime, timezone


RECORDING_TYPE_MAPPING = {
    'Calibration': 'Calib',
    'RestingState': 'RestS',
    'Online': 'OnRun'}


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
    raw.info['experimenter'] = _check_experimenter(experimenter)


def _check_experimenter(experimenter):
    """Checks that the experimenter is a string."""
    assert isinstance(experimenter, str)
    return experimenter


def _add_measurement_date(raw):
    """Add measurement date information to raw instance."""
    fname = Path(raw.filenames[0])
    recording_type = fname.parent.name
    recording_run = fname.name.split('-')[0]

    logs = fname.parent.parent / 'logs.txt'
    with open(logs, 'r') as f:
        lines = f.readlines()
    lines = [line.split(' - ') for line in lines
             if len(line.split(' - ')) > 1]
    logs = [(datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M"),
             line[1].strip(), line[2].strip())
            for line in lines]

    datetime_ = None
    for log in logs:
        conditions = (log[1] == RECORDING_TYPE_MAPPING[recording_type],
                      int(log[2][-1]) == int(recording_run))
        if all(conditions):
            datetime_ = log[0]
            break
    assert datetime_ is not None

    raw.info['meas_date'] = datetime_.astimezone(timezone.utc)


def _add_subject_info(raw, subject, sex, birthday):
    """Add subject information to raw instance."""
    raw.info['subject_info'] = dict()
    # subject ID
    subject = _check_subject(subject, raw)
    if subject is not None:
        raw.info['subject_info']['id'] = subject
        raw.info['subject_info']['his_id'] = str(subject).zfill(3)
    # subject sex - (0, 1, 2) for (Unknown, Male, Female)
    raw.info['subject_info']['sex'] = _check_sex(sex)
    # birthday
    raw.info['subject_info']['birthday'] = _check_birthday(birthday)


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
    try:
        sex = int(sex)
        assert sex in (1, 2)
    except Exception:
        sex = 0
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