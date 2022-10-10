import re
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..io.model import load_session_weights
from ..model import compute_average
from ..utils._checks import _check_path, _check_type
from ..utils._docs import fill_doc


@fill_doc
def apply_weights(
    df: pd.DataFrame, weights: pd.Series, *, copy: bool = False
) -> pd.DataFrame:
    """Apply the given weights mask to the dataframe.

    Parameters
    ----------
    %(df_bp)s
    weights : Series
        Weights used during the online neurofeedback (1 per channel, bads
        included).
    %(copy)s

    Returns
    -------
    %(df_bp)s
    """
    _check_type(df, (pd.DataFrame,), item_name="df")
    _check_type(weights, (pd.Series,), item_name="weights")
    _check_type(copy, (bool,), item_name="copy")
    df = df.copy() if copy else df
    ch_names = [
        col
        for col in df.columns
        if col not in ("participant", "session", "run", "phase", "idx")
    ]
    df.loc[:, ch_names] = df[ch_names] * weights[ch_names].values
    return df


@fill_doc
def apply_group_avg_weights(
    df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    folder: Optional[Union[str, Path]] = None,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """Apply the group-average weights mask to the dataframe.

    The group-average is computed by average all sessions and all participants.

    Parameters
    ----------
    %(df_bp)s
    weights : dict | None
        Weights used during the online neurofeedback (1 per channel, bads
        included).
    folder : path-like | None
        Path to the directory containing raw data with recordings, logs and
        models.
    %(copy)s

    Returns
    -------
    %(df_bp)s

    Notes
    -----
    Only one of weights or folder must be provided. If weights is set to None,
    the average weights are computed from all the participants found in folder.
    """
    _check_type(df, (pd.DataFrame,), item_name="df")
    _check_type(weights, (pd.Series, None), item_name="weights")
    if folder is not None:
        folder = _check_path(folder, "folder", must_exist=True)
    _check_type(copy, (bool,), item_name="copy")
    assert not (weights is None and folder is None)
    assert not (weights is not None and folder is not None)

    if weights is None:
        pattern = re.compile(r"(\d{3})")
        participants = [
            int(p.name) for p in folder.iterdir() if pattern.match(p.name)
        ]
        weights = compute_average(folder, participants)

    return apply_weights(df, weights, copy=copy)


@fill_doc
def apply_participant_avg_weights(
    df: pd.DataFrame,
    weights: Optional[Dict[int, pd.Series]] = None,
    folder: Optional[Union[str, Path]] = None,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """Apply the subject-average weights mask to the dataframe.

    The subject-average is computed by average all sessions for a given
    participants.

    Parameters
    ----------
    %(df_bp)s
    weights : Series | None
        Weights used during the online neurofeedback (1 per channel, bads
        included). The weights are provided as a dictionary with:
            - key : int
                participant IDx
            - value : Series
                weights
    folder : path-like | None
        Path to the directory containing raw data with recordings, logs and
        models.
    %(copy)s

    Returns
    -------
    %(df_bp)s

    Notes
    -----
    Only one of weights or folder must be provided. If weights is set to None,
    the average weights are computed from all the participants found in folder.
    """
    _check_type(df, (pd.DataFrame,), item_name="df")
    _check_type(weights, (dict, None), item_name="weights")
    if weights is not None:
        for key, value in weights.items():
            _check_type(key, ("int",), "participant")
            _check_type(value, (pd.Series,), "weights")
    if folder is not None:
        folder = _check_path(folder, "folder", must_exist=True)
    _check_type(copy, (bool,), item_name="copy")
    assert not (weights is None and folder is None)
    assert not (weights is not None and folder is not None)
    df = df.copy() if copy else df

    if weights is None:
        pattern = re.compile(r"(\d{3})")
        participants = [
            int(p.name) for p in folder.iterdir() if pattern.match(p.name)
        ]
        assert all(elt in participants for elt in df["participant"].unique())

        weights = dict()
        for participant in participants:
            weights[participant] = compute_average(folder, participant)

    ch_names = [
        col
        for col in df.columns
        if col not in ("participant", "session", "run", "phase", "idx")
    ]
    for participant, weights_series in weights.items():
        df.loc[df["participant"] == participant, ch_names] = (
            df.loc[df["participant"] == participant, ch_names]
            * weights_series[ch_names].values
        )
    return df


@fill_doc
def apply_session_weights(
    df: pd.DataFrame, folder: Union[str, Path], *, copy: bool = False
) -> pd.DataFrame:
    """Apply the weights used during a given session to the dataframe.

    Parameters
    ----------
    %(df_bp)s
    %(folder_raw_data)s
    %(copy)s

    Returns
    -------
    %(df_bp)s
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

        df.loc[index, ch_names] = row[ch_names] * weights[ch_names].values
    return df
