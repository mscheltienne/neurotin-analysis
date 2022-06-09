import pandas as pd


def read_csv_evamed(csv):
    """Read the CSV file retrieved from evamed.

    Parameters
    ----------
    csv : path-like
        Path to the .csv file to read.

    Returns
    -------
    df : DataFrame
    """
    df = pd.read_csv(csv, encoding="latin1", skiprows=0, header=1)
    df = df.drop(df.columns[-1], axis=1)
    return df
