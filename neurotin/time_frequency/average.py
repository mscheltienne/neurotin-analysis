from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def add_average_column(df, *, copy: bool = False):
    """Add a column averaging the power on all channels.

    Parameters
    ----------
    %(df_bp)s
    %(copy)s

    Returns
    -------
    %(df_bp)s
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
