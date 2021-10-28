from pathlib import Path
from configparser import ConfigParser

import pandas as pd


def read_paths(ini_file=None):
    """
    Read .ini file containing dataframes .csv paths with ConfigParser.

    Parameters
    ----------
    ini_file : str | Path | None
        Path to the .ini file.

    Returns
    -------
    paths : dict
        Contains the path of the different .csv dataframes.
    """
    ini_file = _check_ini_file(ini_file)
    config = ConfigParser(inline_comment_prefixes=('#', ';'))
    config.optionxform = str
    config.read(ini_file)
    paths = {key.replace('_', '-').lower(): path
             for key, path in config.items('paths')}
    return paths


def _check_ini_file(ini_file):
    """Check argument ini_file."""
    if ini_file is None:
        ini_file = Path(__file__).parent/'paths.ini'
    else:
        ini_file = Path(ini_file)
    assert ini_file.exists(), 'could not find %s' % ini_file
    assert ini_file.suffix == '.ini', \
        "file suffix %s is not .ini" % ini_file.suffix
    return str(ini_file)


def read_csv(file):
    """
    Read .csv returned by Evamed. The encoding is 'latin1'.

    Parameters
    ----------
    file : str | Path
        Path to the .csv file exported from Evamed

    Returns
    -------
    df : DataFrame
        Loaded pandas DataFrame.
    """
    file = _check_file(file)
    df = pd.read_csv(file, encoding='latin1', skiprows=0, header=1)
    df = df.drop(df.columns[-1], axis=1)
    return df


def _check_file(file):
    """Check argument file."""
    file = Path(file)
    assert file.exists(), 'could not find %s' % file
    assert file.suffix == '.csv', "file suffix %s is not .csv" % file.suffix
    return file
