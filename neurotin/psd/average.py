import numpy as np
from scipy.stats import zscore

from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def add_average_column(df, *, copy: bool = False):
    """Add a column averaging the power on all channels.

    Parameters
    ----------
    %(df_psd)s
        An 'avg' column is added averaging the power on all channels.
    %(copy)s

    Returns
    -------
    %(df_psd)s
        The average power across channels has been added in the column 'avg'.
    """
    _check_type(copy, (bool,), item_name="copy")
    df = df.copy() if copy else df

    ch_names = [
        col
        for col in df.columns
        if col not in ("participant", "session", "run", "phase", "idx")
    ]
    df["avg"] = df[ch_names].mean(axis=1)
    return df


@fill_doc
def remove_outliers(df, score: float = 2.0, *, copy: bool = False):
    """Remove outliers from the average column.

    Parameters
    ----------
    %(df_psd)s
        An 'avg' column is added averaging the power on all channels if it is
        not present.
    score : float
        ZScore threshold applied on each participant/session/run to eliminate
        outliers.
    %(copy)s

    Returns
    -------
    %(df_psd)s
        Outliers have been removed.
    """
    _check_type(score, ("numeric",), item_name="score")
    _check_type(copy, (bool,), item_name="copy")
    df = df.copy() if copy else df
    if "avg" not in df.columns:
        df = add_average_column(df)

    outliers_idx = list()
    participants = sorted(df["participant"].unique())
    for participant in participants:
        df_participant = df[df["participant"] == participant]

        sessions = sorted(df_participant["session"].unique())
        for session in sessions:
            df_session = df_participant[df_participant["session"] == session]

            runs = sorted(df_session["run"].unique())
            for run in runs:
                df_run = df_session[df_session["run"] == run]

                # search for outliers and retrieve index
                outliers = df_run[~(np.abs(zscore(df_run["avg"])) <= score)]
                outliers_idx.extend(list(outliers.index))

    df.drop(index=outliers_idx, inplace=True)
    return df
