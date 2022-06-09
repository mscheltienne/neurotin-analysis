import pickle
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from mne.io import Info
from numpy.typing import NDArray

from ..utils._checks import (
    _check_participant,
    _check_path,
    _check_session,
    _check_type,
)
from ..utils._docs import fill_doc


@fill_doc
def load_model(
    folder, participant: int, session: int, model_idx: Union[int, str] = "auto"
) -> Tuple[NDArray[float], Info, Dict[str, float], NDArray[float], int]:
    """Load a saved model for a given participant and session.

    Parameters
    ----------
    %(folder_data)s
    %(participant)s
    %(session)s
    model_idx : int | 'auto'
        ID of the model to load. If 'auto', the latest model is loaded.

    Returns
    -------
    weights : array
        Array of weights between 0 and 1 of shape (64 channels, )
    info : Info
        MNE measurement info instance.
    reject : dict
        Global peak-to-peak rejection threshold. Only key should be 'eeg'.
    reject_local : array
        Array of local peak-to-peak rejection threshold of shape shape
        (64 channels, ).
    calib_idx : int
        ID of the calibration used to generate the model.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    participant = _check_participant(participant)
    participant_folder = str(participant).zfill(3)
    session = _check_session(session)
    model_idx = _check_model_idx(model_idx)

    session_dir = folder / participant_folder / f"Session {session}"

    if model_idx == "auto":
        # read_logs
        logs = [log for log in _read_logs(session_dir) if log[1] == "Model"]
        valid_model_idx = [
            int(log[2].split(" ")[2]) for log in logs if len(log) == 3
        ]
        assert 0 < len(valid_model_idx)
        model_idx = max(valid_model_idx)

    model_fname = session_dir / "Model" / f"{model_idx}-model.pcl"
    with open(model_fname, "rb") as f:
        weights, info, reject, reject_local, calib_idx = pickle.load(f)

    return weights, info, reject, reject_local, calib_idx


def _check_model_idx(model_idx: Union[int, str]) -> Union[int, str]:
    """Check argument model_idx."""
    _check_type(model_idx, ("int", str), item_name="model_idx")
    if isinstance(model_idx, str):
        model_idx = model_idx.lower().strip()
        assert model_idx == "auto", "Invalid model ID."
    else:
        assert 1 <= model_idx, "Invalid model ID."
    return model_idx


def _read_logs(session_dir):
    """Read logs for a given participant/session."""
    session_dir = _check_path(
        session_dir, item_name="session_dir", must_exist=True
    )
    logs_file = _check_path(
        session_dir / "logs.txt", item_name="logs_file", must_exist=True
    )
    with open(logs_file, "r") as f:
        lines = f.readlines()
    lines = [line.split(" - ") for line in lines if len(line.split(" - ")) > 1]
    logs = [
        [datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M")]
        + [line[k].strip() for k in range(1, len(line))]
        for line in lines
    ]
    return sorted(logs, key=lambda x: x[0], reverse=False)


@fill_doc
def load_session_weights(
    folder, participant: int, session: int, replace_bad_with=0
):
    """Load the weights used during a session and return them as a Dataframe.

    Parameters
    ----------
    %(folder_data)s
    %(participant)s
    %(session)s
    replace_bad_with : float | np.nan
        What is used to fill bad channel values in the DataFrame.

    Returns
    -------
    weights : DataFrame
        Weights used during the online neurofeedback (1 per channel, bads
        included).
    """
    replace_bad_with = _check_type(
        replace_bad_with, ("numeric", np.nan), item_name="replace_bad_with"
    )
    weights, info, _, _, _ = load_model(folder, participant, session)

    # weights is missing the channels set as bads in info
    weights_with_bads = list()
    idx = 0
    for ch in info.ch_names:
        if ch in info["bads"]:
            weights_with_bads.append(replace_bad_with)
        else:
            weights_with_bads.append(weights[idx])
            idx += 1

    return pd.DataFrame.from_dict(
        {"channel": info.ch_names, "weight": weights_with_bads},
        orient="columns",
    )
