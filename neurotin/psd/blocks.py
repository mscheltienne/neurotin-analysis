from itertools import product

import numpy as np
import pandas as pd

from ..utils._checks import _check_type
from ..utils._docs import fill_doc


# TODO: column should be str | list | tuple -> is str it gets converted into a
# list. If 'all', the list becomes all columns. A single pipeline should exist
# no matter the number of columns.
@fill_doc
def blocks_difference_between_consecutive_phases(df, column="avg"):
    """
    Compute the difference between a column in a regulation phase and in the
    preceding non-regulation phase.

    Parameters
    ----------
    %(df_psd)s
    column : str
        Label of the column on which the difference is computed between both
        phases. Can also be 'all' to include all columns.

    Returns
    -------
    df : DataFrame
        Difference between the average band power in the regulation phase and
        the preceding non-regulation phase.
            participant : int - Participant ID
            session : int - Session ID (1 to 15)
            run : int - Run ID
            idx : ID of the phase within the run (0 to 9)
            diff or col-diff : float - PSD difference (regulation - rest)
    """
    _check_type(column, (str,), item_name="column")
    if column != "all":
        assert column in df.columns

    # check keys
    keys = ["participant", "session", "run", "idx"]
    assert all(key in df.columns for key in keys)

    if column == "all":
        data = _blocks_difference_between_consecutive_phases_all_columns(df)
    else:
        data = _blocks_difference_between_consecutive_phases_single_column(
            df, column
        )

    return pd.DataFrame.from_dict(data, orient="columns")


def _blocks_difference_between_consecutive_phases_single_column(df, column):
    """
    Compute the difference between consecutive phases from a single column.
    """
    # container for new df with diff between phases
    keys = ["participant", "session", "run", "idx"]
    data = {key: [] for key in keys + ["diff"]}

    participants = sorted(df["participant"].unique())
    for participant in participants:
        df_participant = df[df["participant"] == participant]

        sessions = sorted(df_participant["session"].unique())
        for session in sessions:
            df_session = df_participant[df_participant["session"] == session]

            runs = sorted(df_session["run"].unique())
            for run in runs:
                df_run = df_session[df_session["run"] == run]

                index = sorted(df_session["idx"].unique())
                for idx in index:
                    df_idx = df_run[df_run["idx"] == idx]

                    # compute the difference between regulation and rest
                    reg = df_idx[df_idx.phase == "regulation"][column]
                    non_reg = df_idx[df_idx.phase == "non-regulation"][column]
                    try:
                        diff = reg.values[0] - non_reg.values[0]
                    except IndexError:
                        continue

                    # fill dict
                    data["participant"].append(participant)
                    data["session"].append(session)
                    data["run"].append(run)
                    data["idx"].append(idx)
                    data["diff"].append(diff)

    return data


def _blocks_difference_between_consecutive_phases_all_columns(df):
    """
    Compute the difference between consecutive phases for all columns.
    """
    # container for new df with diff between phases
    keys = ["participant", "session", "run", "idx"]
    columns = [col for col in df.columns if col not in keys + ["phase"]]
    data = {key: [] for key in keys + [col + "-diff" for col in columns]}
    participants = sorted(df["participant"].unique())
    for participant in participants:
        df_participant = df[df["participant"] == participant]

        sessions = sorted(df_participant["session"].unique())
        for session in sessions:
            df_session = df_participant[df_participant["session"] == session]

            runs = sorted(df_session["run"].unique())
            for run in runs:
                df_run = df_session[df_session["run"] == run]

                index = sorted(df_session["idx"].unique())
                for idx in index:
                    df_idx = df_run[df_run["idx"] == idx]

                    # compute the difference between regulation and rest
                    reg = df_idx[df_idx.phase == "regulation"][columns]
                    non_reg = df_idx[df_idx.phase == "non-regulation"][columns]
                    try:
                        diff = reg.values[0, :] - non_reg.values[0, :]
                    except IndexError:
                        continue

                    # fill dict
                    data["participant"].append(participant)
                    data["session"].append(session)
                    data["run"].append(run)
                    data["idx"].append(idx)
                    for k, col in enumerate(columns):
                        data[col + "-diff"].append(diff[k])

    return data


def blocks_count_success(df, group_session: bool = False):
    """
    Count the positive/negative diff values by session.
    The count is normalized by the number of observations.

    Parameters
    ----------
    df : DataFrame
        Difference between the average band power in the regulation phase and
        the preceding non-regulation phase.
            participant : int - Participant ID
            session : int - Session ID (1 to 15)
            run : int - Run ID
            idx : ID of the phase within the run (0 to 9)
            diff : float - PSD difference (regulation - rest)
    group_sessions : bool
        If True, all session are grouped together.

    Returns
    -------
    df_positives : DataFrame
        Counts of positives 'diff'.
    df_negatives : DataFrame
        Counts of negatives 'diff'.
    """
    assert "diff" in df.columns

    # reset order
    df = df.sort_values(
        by=["participant", "session", "run", "idx"], ascending=True
    )
    df.reset_index()

    # check sign
    df["sign"] = np.sign(df["diff"])
    # groupby
    by = ["participant"] if group_session else ["participant", "session"]
    counts = df.groupby(by=by, dropna=True)["sign"].value_counts()

    participants = df["participant"].unique()
    sessions = df["session"].unique()

    if group_session:
        positives, negatives = _blocks_count_success_group_session(
            counts, participants
        )
    else:
        positives, negatives = _blocks_count_success(
            counts, participants, sessions
        )

    # create df
    df_positives = pd.DataFrame.from_dict(positives, orient="columns")
    df_negatives = pd.DataFrame.from_dict(negatives, orient="columns")

    # reset order
    by = ["participant"] if group_session else ["participant", "session"]
    df_positives.sort_values(by=by, ascending=True, inplace=True)
    df_positives.reset_index()
    df_negatives.sort_values(by=by, ascending=True, inplace=True)
    df_negatives.reset_index()

    return df_positives, df_negatives


def _blocks_count_success(counts, participants, sessions):
    """Counts success for each participant/session individually."""
    positives = {key: [] for key in ("participant", "session", "count")}
    negatives = {key: [] for key in ("participant", "session", "count")}

    for participant, session in product(participants, sessions):
        try:
            pos = counts[participant, session, 1]
            neg = counts[participant, session, -1]
        except KeyError:
            pos = np.nan
            neg = np.nan

        # add common data to dict
        positives["participant"].append(participant)
        positives["session"].append(session)
        negatives["participant"].append(participant)
        negatives["session"].append(session)

        if any(np.isnan(x) or x == 0 for x in (pos, neg)):
            positives["count"].append(np.nan)
            negatives["count"].append(np.nan)
        else:
            # normalize
            total = pos + neg
            pos = pos / total
            neg = neg / total
            # add to dict
            positives["count"].append(pos)
            negatives["count"].append(-neg)

    return positives, negatives


def _blocks_count_success_group_session(counts, participants):
    """Counts success for each participant by grouping sessions."""
    positives = {key: [] for key in ("participant", "count")}
    negatives = {key: [] for key in ("participant", "count")}

    for participant in participants:
        try:
            pos = counts[participant, 1]
            neg = counts[participant, -1]
        except KeyError:
            pos = np.nan
            neg = np.nan

        # add common data to dict
        positives["participant"].append(participant)
        negatives["participant"].append(participant)

        if any(np.isnan(x) or x == 0 for x in (pos, neg)):
            positives["count"].append(np.nan)
            negatives["count"].append(np.nan)
        else:
            # normalize
            total = pos + neg
            pos = pos / total
            neg = neg / total
            # add to dict
            positives["count"].append(pos)
            negatives["count"].append(-neg)

    return positives, negatives
