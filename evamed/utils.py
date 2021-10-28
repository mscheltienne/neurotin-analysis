from pathlib import Path
from configparser import ConfigParser

import pandas as pd


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
    file = Path(file)
    assert file.exists(), 'could not find %s' % file
    assert file.suffix == '.csv', "file suffix %s is not .csv" % file.suffix
    df = pd.read_csv(file, encoding='latin1', skiprows=0, header=1)
    df = df.drop(df.columns[-1], axis=1)
    return df


def read_paths(ini_file):
    """
    Read .ini file containing dataframes .csv paths with ConfigParser.

    Parameters
    ----------
    ini_file : str | Path
        Path to the .ini file.

    Returns
    -------
    paths : dict
        Contains the path of the different .csv dataframes.
    """
    ini_file = Path(ini_file)
    assert ini_file.exists(), 'could not find %s' % ini_file
    assert ini_file.suffix == '.ini', \
        "file suffix %s is not .ini" % ini_file.suffix
    config = ConfigParser(inline_comment_prefixes=('#', ';'))
    config.optionxform = str
    config.read(str(ini_file))
    paths = {key.replace('_', '-').lower(): path
             for key, path in config.items('paths')}
    return paths
