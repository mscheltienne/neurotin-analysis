from typing import List, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..utils._checks import (
    _check_participant,
    _check_participants,
    _check_type,
)
from ..utils._docs import fill_doc


@fill_doc
def boxplot_scores_evolution(
    csv,
    participant: Union[int, list, tuple],
    scores: int = 10,
    swarmplot: bool = False,
    figsize: Tuple[float, float] = (10.0, 5.0),
):
    """Plot the NFB scores as boxplots.

    X: Session
    Y: Score
    Hue: Score ID (0 to 10)

    The NFB scores displayed are logged in a .csv file with the syntax:
        [participant, session, model_idx, online_idx, transfer, scores [...]]

    The evolution of the NFB score during the 15 sessions is plotted for the
    given participant with boxplots. Scores from different part of the NFB runs
    can be displayed by providing the argument scores. By default, the last
    score corresponding to the total score obtained on a given run is used.

    Parameters
    ----------
    csv : path-like
        Path to the 'scores_logs.csv' file to read.
    %(participant)s
    scores : int | list of int
        ID of the non-regulation/regulation cycle score to include, or list
        of the IDs to include. Each cycle is displayed as a separate boxplot.
        Must be between 1 and 10 included.
    swarmplot : bool, optional
        If True, plots the datapoints on top of the boxes with a swarmplot.
    %(plt.figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_participant(participant)
    scores = _check_scores_idx(scores)
    _check_type(swarmplot, (bool,), item_name="swarmplot")
    _check_type(figsize, (tuple,), item_name="figsize")

    # Select data
    df = pd.read_csv(csv)
    df = df.loc[df["Participant"] == int(participant)]
    df = pd.melt(
        df,
        id_vars="Session",
        value_vars=[f"Score {k}" for k in scores],
        var_name="Score ID",
        value_name="Score",
    )

    # Plot
    f, ax = plt.subplots(1, 1, figsize=tuple(figsize))
    sns.boxplot(
        x="Session", y="Score", hue="Score ID", data=df, palette="muted", ax=ax
    )
    if swarmplot:
        sns.swarmplot(
            x="Session",
            y="Score",
            hue="Score ID",
            data=df,
            size=3,
            color="black",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[: len(scores)], labels=labels[: len(scores)])

    return f, ax


@fill_doc
def boxplot_scores_between_participants(
    csv,
    participants: Union[int, list, tuple],
    scores: int = 10,
    swarmplot: bool = False,
    figsize: Tuple[float, float] = (10.0, 5.0),
):
    """Plot the NFB scores as boxplots.

    X: Participant
    Y: Score
    Hue: Score ID (0 to 10)

    The NFB scores displayed are logged in a .csv file with the syntax:
        [participant, session, model_idx, online_idx, transfer, scores [...]]

    The scores obtained during the 15 sessions are plotted in a single
    boxplot for each participant.

    Parameters
    ----------
    csv : path-like
        Path to the 'scores_logs.csv' file to read.
    %(participant)s
    scores : int | list of int
        ID of the non-regulation/regulation cycle score to include, or list
        of the IDs to include. Each cycle is displayed as a separate boxplot.
        Must be between 1 and 10 included.
    swarmplot : bool, optional
        If True, plots the datapoints on top of the boxes with a swarmplot.
    %(plt.figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    participants = _check_participants(participants)
    scores = _check_scores_idx(scores)
    _check_type(swarmplot, (bool,), item_name="swarmplot")
    _check_type(figsize, (tuple,), item_name="figsize")

    # Select data
    df = pd.read_csv(csv)
    df = pd.melt(
        df,
        id_vars="Participant",
        value_vars=[f"Score {k}" for k in scores],
        var_name="Score ID",
        value_name="Score",
    )
    df = df[df["Participant"].isin(participants)]

    # Plot
    f, ax = plt.subplots(1, 1, figsize=tuple(figsize))
    sns.boxplot(
        x="Participant",
        y="Score",
        hue="Score ID",
        data=df,
        palette="muted",
        ax=ax,
    )
    if swarmplot:
        sns.swarmplot(
            x="Participant",
            y="Score",
            hue="Score ID",
            data=df,
            size=3,
            color="black",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[: len(scores)], labels=labels[: len(scores)])

    return f, ax


def _check_scores_idx(scores: Union[int, list, tuple]) -> List[int]:
    """Check that the scores passed are valid."""
    _check_type(scores, ("int", list, tuple), item_name="scores")
    if isinstance(scores, int):
        scores = [scores]
    elif isinstance(scores, tuple):
        scores = list(scores)
    for score in scores:
        _check_type(score, ("int",), item_name="score")
    assert all(1 <= score <= 10 for score in scores)
    return scores
