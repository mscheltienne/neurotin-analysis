from pathlib import Path

import pandas as pd


def read_csv(file):
    """
    Read .csv returned by Evamed. The encoding is 'latin1'.

    Parameters
    ----------
    file : str | Path
        Path to the csv file exported from Evamed

    Returns
    -------
    df : DataFrame
        Loaded pandas DataFrame.
    """
    file = Path(file)
    assert file.exists(), 'could not find %s' % file
    assert file.suffix == '.csv', "file suffix %s is not .csv" % file.suffix
    return pd.read_csv(file, encoding='latin1')
