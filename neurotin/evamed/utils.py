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
    df = pd.read_csv(file, encoding='latin1', skiprows=0, header=1)
    df = df.drop(df.columns[-1], axis=1)
    return df
