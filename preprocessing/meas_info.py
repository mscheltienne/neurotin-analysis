from pathlib import Path
from datetime import datetime


RECORDING_TYPE_MAPPING = {
    'Calibration': 'Calib',
    'RestingState': 'RestS',
    'Online': 'OnRun'}


def fill_info(raw):
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

    Returns
    -------
    raw : Raw instance modified in-place.
    """
    fname = Path(raw.filenames[0])

    # Description
    subject = int(fname.parent.parent.parent.name)
    session = int(fname.parent.parent.name.split()[-1])
    recording_type = fname.parent.name
    recording_run = fname.name.split('-')[0]
    raw.info['description'] = f'Subject {subject} - Session {session} '+\
                              f'- {recording_type} {recording_run}'

    # Device info
    raw.info['device_info'] = dict()
    raw.info['device_info']['type'] = 'EEG'
    raw.info['device_info']['model'] = 'eego mylab'
    serial = fname.stem.split('-raw')[0].split('-')[-1].split()[1]
    raw.info['device_info']['serial'] = serial
    raw.info['device_info']['site'] = \
        'https://www.ant-neuro.com/products/eego_mylab'

    # Experimenter
    raw.info['experimenter'] = 'Mathieu Scheltienne'

    # Measurement datetime
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
    raw.info['meas_date'] = datetime_

    return raw
