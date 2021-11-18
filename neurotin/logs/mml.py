"""
The MML are logged with pandas in a .csv file with the syntax:
    [participant, session, MML volume]
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from . import _check_participants
from ..io.csv import read_csv
from ..utils.checks import _check_type


def plot_mml_across_participants(csv, participants, figsize=(10, 5)):
    """
    Plot the MML evolution for the given participants.

    Parameters
    ----------
    csv : str | pathlib.Path
        Path to the csv file to read. Must be in .csv format.
    participants : list of int
        List of participants ID (int) to include.
    figsize : tuple, optional
        Matplotlib figure's size. The default is (10, 5).

    Returns
    -------
    f : Figure
    ax : axes.Axes
    """
    participants = _check_participants(participants)
    figsize = _check_type(figsize, (tuple, ), item_name='figsize')

    # Select data
    df = read_csv(csv)
    df = pd.melt(
        df, id_vars=('Participant', 'Session'), value_vars='MML Volume',
        var_name='MML', value_name='Volume')
    df = df[df['Participant'].isin(participants)]
    df.drop_duplicates(
        subset=('Participant', 'Session'), keep='last', inplace=True)

    # Plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(
        x='Session', y='Volume', hue='Participant', data=df,
        palette='muted', style="Participant", markers=True, dashes=False)
    ax.set_xticks(range(1, 16, 1))
    ax.set_yticks(np.arange(0, max(df.Volume)+2.5, 2.5))

    return f, ax