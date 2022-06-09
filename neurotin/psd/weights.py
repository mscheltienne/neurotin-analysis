import pandas as pd

from ..io.model import load_session_weights
from ..utils._checks import _check_path, _check_type
from ..utils._docs import fill_doc


@fill_doc
def weights_apply_mask(df, weights, *, copy: bool = False):
    """Apply the weights mask to the PSD dataframe.

    Parameters
    ----------
    %(df_psd)s
    weights : DataFrame
        Weights used during the online neurofeedback (1 per channel, bads
        included).
    %(copy)s

    Returns
    -------
    %(df_psd)s
    """
    _check_type(df, (pd.DataFrame,), item_name="df")
    _check_type(weights, (pd.DataFrame,), item_name="weights")
    _check_type(copy, (bool,), item_name="copy")
    df = df.copy() if copy else df

    ch_names = weights["channel"]
    df.loc[:, ch_names] = df[ch_names] * weights["weight"].values
    return df


@fill_doc
def weights_apply_session_mask(df, folder, *, copy: bool = False):
    """Apply the weights used during a given session to the PSD dataframe.

    Parameters
    ----------
    %(df_psd)s
    %(folder_data)s
    %(copy)s

    Returns
    -------
    %(df_psd)s
    """
    _check_type(df, (pd.DataFrame,), item_name="df")
    folder = _check_path(folder, "folder", must_exist=True)
    _check_type(copy, (bool,), item_name="copy")
    df = df.copy() if copy else df

    participant_session = None
    for index, row in df.iterrows():
        # load weights if needed
        if participant_session != (row["participant"], row["session"]):
            weights = load_session_weights(
                folder, row["participant"], row["session"]
            )
            participant_session = (row["participant"], row["session"])
            ch_names = weights["channel"]

        df.loc[index, ch_names] = row[ch_names] * weights["weight"].values
    return df
