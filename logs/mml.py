"""
The MML are logged with pandas in a .csv file with the syntax:
    [participant, session, MML volume]
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import read_csv, _check_participants


def plot_mml_evolution_per_participants(csv, participants, figsize=(10, 5)):
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

    # Select data
    df = read_csv(csv)
    df = pd.melt(
        df, id_vars=('Participant', 'Session'), value_vars='MML Volume',
        var_name='MML', value_name='Volume')
    df = df[df['Participant'].isin(participants)]

    # Plot
    f, ax = plt.subplots(1, 1, figsize=tuple(figsize))
    sns.lineplot(
        x='Session', y='Volume', hue='Participant', data=df,
        palette='muted', style="Participant", markers=True, dashes=False)

    return f, ax


#%% Main
if __name__ == '__main__':
    csv = r'/Volumes/NeuroTin-EEG/Data/mml_logs.csv'
    plot_mml_evolution_per_participants(csv, participants=(60, 61, 65))
