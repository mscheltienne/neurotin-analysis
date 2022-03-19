import pandas as pd

from ..utils._checks import _check_path


def read_csv(csv, **kwargs):
    """
    Read the CSV file and returns the stored pandas DataFrame.

    Parameters
    ----------
    csv : path-like
        Path to the .csv file to read.
    kwargs are passed to pd.read_csv().

    Returns
    -------
    df : DataFrame
    """
    csv = _check_path(csv, item_name='csv', must_exist=True)
    assert csv.suffix == '.csv', 'Provided file is not a .csv file.'
    df = pd.read_csv(csv, **kwargs)
    return df


def read_csv_evamed(csv):
    """
    Read the CSV file retrieved from evamed.

    Parameters
    ----------
    csv : path-like
        Path to the .csv file to read.

    Returns
    -------
    df : DataFrame
    """
    df = read_csv(csv, encoding='latin1', skiprows=0, header=1)
    df = df.drop(df.columns[-1], axis=1)
    return df
