import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .. import logger
from ..io.model import load_session_weights
from ..utils._checks import _check_participants, _check_path
from ..utils._docs import fill_doc


@fill_doc
def compute_average(
    folder: Union[str, Path],
    participants: Union[int, List[int], Tuple[int, ...]],
):
    """Compute the average model across all session and all participants.

    Parameters
    ----------
    %(folder_raw_data)s
    %(participants)s

    Returns
    -------
    df : DataFrame
        Average weight per channel.
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    participants = _check_participants(participants)

    # Load all models into a DataFrame
    df = None
    for participant in participants:
        # look for sessions
        pattern = re.compile(r"Session (\d{1,2})")
        folder_ = folder / str(participant).zfill(3)
        sessions = [
            int(session)
            for path in folder_.iterdir()
            for session in re.findall(pattern, str(path))
        ]

        for session in sessions:
            try:
                weights = load_session_weights(
                    folder, participant, session, replace_bad_with=np.nan
                )
            except FileNotFoundError:
                logger.warning(
                    "Model for participant %s and session %s not found.",
                    participant,
                    session,
                )
                continue
            weights.rename(
                columns={"weight": f"{participant}-S{session}"}, inplace=True
            )
            weights.set_index("channel", inplace=True)

            # concatenate
            df = weights if df is None else pd.concat([df, weights], axis=1)

    return df.mean(axis=1, skipna=True)
