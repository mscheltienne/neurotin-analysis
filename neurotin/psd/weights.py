import pandas as pd

from ..io.model import load_session_weights
from ..utils._checks import _check_type, _check_path
from ..utils._docs import fill_doc


@fill_doc
def apply_weights_mask(df, weights, *, copy=False):
    """
    Apply the weights mask to the PSD dataframe.

    Parameters
    ----------
    %(df_psd)s
    %(df_weights)s
    %(copy)s

    Returns
    -------
    %(df_psd)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    _check_type(weights, (pd.DataFrame, ), item_name='weights')
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df

    ch_names = weights['channel']
    df.loc[:, ch_names] = df[ch_names] * weights['weight'].values
    return df


@fill_doc
def apply_weights_session(df, folder, *, copy=False):
    """
    Apply the weights used during a given session to the PSD dataframe.

    Parameters
    ----------
    %(df_psd)s
    %(folder_data)s
    %(copy)s

    Returns
    -------
    %(df_psd)s
    """
    _check_type(df, (pd.DataFrame, ), item_name='df')
    folder = _check_path(folder, 'folder', must_exist=True)
    _check_type(copy, (bool, ), item_name='copy')
    df = df.copy() if copy else df

    participant_session = None
    for index, row in df.iterrows():
        # load weights if needed
        if participant_session != (row['participant'], row['session']):
            weights = load_session_weights(
                folder, row['participant'], row['session'])
            participant_session = (row['participant'], row['session'])
            ch_names = weights['channel']

        df.loc[index, ch_names] = row[ch_names] * weights['weight'].values
    return df
