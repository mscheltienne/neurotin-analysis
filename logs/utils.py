from pathlib import Path

import pandas as pd


def read_csv(csv):
    """
    Read the CSV file and returns the stored pandas dataframe.

    Parameters
    ----------
    csv : str | pathlib.Path
        Path to the csv file to read. Must be in .csv format.

    Returns
    -------
    df : pandas.DataFrame
    """
    csv = Path(csv)
    if csv.suffix != '.csv':
        raise IOError('Provided file is not a CSV file.')
    if not csv.exists():
        raise IOError('Provided file does not exists.')
    df = pd.read_csv(csv)
    return df


def _check_participants(participants):
    if isinstance(participants, (int, float)):
        participants = [int(participants)]
    elif isinstance(participants, (list, tuple)):
        participants = [int(participant) for participant in participants]
    else:
        raise TypeError
    return participants
