import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .. import logger
from ..io.model import load_session_weights
from ..utils._checks import _check_participants, _check_path
from ..utils._docs import fill_doc


@fill_doc
def create_weight_dataframe(
    folder: Union[str, Path],
    participants: Union[int, List[int], Tuple[int, ...]],
) -> Dict[int, pd.DataFrame]:
    """Load all the sessions weights in a DataFrame.

    Parameters
    ----------
    %(folder_raw_data)s
    %(participants)s

    Returns
    -------
    dfs : dict of (participant: DataFrames)
    """
    folder = _check_path(folder, item_name="folder", must_exist=True)
    participants = _check_participants(participants)

    dfs = dict()
    desired_index = [f"S{k}" for k in range(1, 16)]
    for participant in participants:
        # look for sessions
        pattern = re.compile(r"Session (\d{1,2})")
        folder_ = folder / str(participant).zfill(3)
        sessions = [
            int(session)
            for path in folder_.iterdir()
            for session in re.findall(pattern, str(path))
        ]

        df = None
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
                columns={"weight": f"S{session}"}, inplace=True
            )
            weights.set_index("channel", inplace=True)

            # concatenate
            df = weights if df is None else pd.concat([df, weights], axis=1)

        new_index = [elt for elt in desired_index if elt in df.columns]
        dfs[participant] = df.reindex(new_index, axis=1)
    return dfs
