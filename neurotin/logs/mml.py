import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..utils._checks import _check_participants, _check_type
from ..utils._docs import fill_doc


@fill_doc
def lineplot_mml_evolution(csv, participants, figsize=(10, 5)):
    """Plot the MML results as lineplot.

    X: Session
    Y: MML
    Hue: Participant

    Minimum Masking Level test are logged in a .csv file with the syntax:
        [participant, session, MML volume]
    The evolution of the minimum masking level during the 15 sessions is
    plotted for each participant as a lineplot.

    Parameters
    ----------
    csv : path-like
        Path to the 'mml_logs.csv' file to read.
    %(participants)s
    %(plt.figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    participants = _check_participants(participants)
    _check_type(figsize, (tuple,), item_name="figsize")

    # Select data
    df = pd.read_csv(csv)
    df = pd.melt(
        df,
        id_vars=("Participant", "Session"),
        value_vars="MML Volume",
        var_name="MML",
        value_name="Volume",
    )
    df = df[df["Participant"].isin(participants)]
    df.drop_duplicates(
        subset=("Participant", "Session"), keep="last", inplace=True
    )

    # Plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(
        x="Session",
        y="Volume",
        hue="Participant",
        data=df,
        palette="muted",
        style="Participant",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_xticks(range(1, 16, 1))
    ax.set_yticks(np.arange(0, max(df.Volume) + 2.5, 2.5))

    return f, ax
