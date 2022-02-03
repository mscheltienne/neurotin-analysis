import numpy as np
import pandas as pd
import re

from .. import logger
from ..io.model import load_session_weights
from ..utils.docs import fill_doc
from ..utils.checks import _check_path, _check_participants


@fill_doc
def compute_average(raw_folder, participants):
    """
    Compute the average model across all session and across all participants.

    Parameters
    ----------
    %(raw_folder)s
    participants : int | list | tuple
        Participant ID or list of participant IDs to merge.

    Returns
    -------
    df : DataFrame
        Average weight per channel.
    """
    raw_folder = _check_path(raw_folder, item_name='folder', must_exist=True)
    participants = _check_participants(participants)

    # Load all models into a DataFrame
    df = None
    for participant in participants:
        # look for sessions
        pattern = re.compile(r'Session (\d{1,2})')
        folder = raw_folder / str(participant).zfill(3)
        sessions = [int(session)
                    for path in folder.iterdir()
                    for session in re.findall(pattern, str(path))]

        for session in sessions:
            try:
                weights = load_session_weights(
                    raw_folder, participant, session, replace_bad_with=np.nan)
            except FileNotFoundError:
                logger.warning(
                    'Model for participant %s and session %s not found.',
                    participant, session)
                continue
            weights.rename(columns={'weight': f'{participant}-S{session}'},
                           inplace=True)
            weights.set_index('channel', inplace=True)

            # concatenate
            df = weights if df is None else pd.concat([df, weights], axis=1)

    return df.mean(axis=1, skipna=True)
