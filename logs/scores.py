"""
The scores are logged with pandas in a .csv file with the syntax:
    [[participant, session, model_idx, online_idx, int(transfer_run)] + scores]
"""
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_csv(csv):
    """
    Read the CSV file and returns the stored pandas dataframe.

    Parameters
    ----------
    csv : str | pathlib.Path
        Path to the csv file to read. Must be in .csv format.

    Returns
    -------
    df : pandas.DataFrame
    """
    csv = Path(csv)
    if csv.suffix != '.csv':
        raise IOError('Provided file is not a CSV file.')
    if not csv.exists():
        raise IOError('Provided file does not exists.')
    df = pd.read_csv(csv)
    return df


def _check_scores_idx(scores):
    if isinstance(scores, (int, float)):
        scores = [int(scores)]
    elif isinstance(scores, (list, tuple)):
        scores = [int(score) for score in scores]
    else:
        raise TypeError
    if not all(1 <= score <= 10 for score in scores):
        raise ValueError
    return scores


def plot_score_evolution_per_participant(
        csv, participant, scores=10, datapoints=False, figsize=(10, 5)):
    """
    Plot as boxplots the score evolution across session for a given
    participant. Multiple score IDs can be plotted simultaneously.

    Parameters
    ----------
    csv : str | pathlib.Path
        Path to the csv file to read. Must be in .csv format.
    participant : int
        Participant ID.
    scores : int | list, optional
        Score ID or list of scores IDs to plot.
        IDs are defined between 0 and 10. The default is 10.
    datapoints : bool, optional
        If True, plots the datapoints on top of the boxes.
        The default is False.
    figsize : tuple, optional
        Matplotlib figure's size. The default is (10, 5).

    Returns
    -------
    f : Figure
    ax : axes.Axes
    """
    scores = _check_scores_idx(scores)

    # Select data
    df = read_csv(csv)
    df = df.loc[df['Participant'] == int(participant)]
    df = pd.melt(
        df, id_vars='Session', value_vars=[f'Score {k}' for k in scores],
        var_name='Score ID', value_name='Score')

    # Plot
    f, ax = plt.subplots(1, 1, figsize=tuple(figsize))
    sns.boxplot(
        x='Session', y='Score', hue='Score ID', data=df,
        palette='muted')
    if datapoints:
        sns.swarmplot(
            x='Session', y='Score', hue='Score ID', data=df,
            size=3, color='black', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:len(scores)], labels=labels[:len(scores)])

    return f, ax


def plot_score_across_participants(
        csv, scores=10, datapoints=False, figsize=(10, 5)):
    """
    Plot as boxplots the score in all session across participants. Multiple
    score IDs can be plotted simultaneously.

    Parameters
    ----------
    csv : str | pathlib.Path
        Path to the csv file to read. Must be in .csv format.
    scores : int | list, optional
        Score ID or list of scores IDs to plot.
        IDs are defined between 0 and 10. The default is 10.
    datapoints : bool, optional
        If True, plots the datapoints on top of the boxes.
        The default is False.
    figsize : tuple, optional
        Matplotlib figure's size. The default is (10, 5).

    Returns
    -------
    f : Figure
    ax : axes.Axes
    """
    scores = _check_scores_idx(scores)

    # Select data
    df = read_csv(csv)
    df = pd.melt(
        df, id_vars='Participant', value_vars=[f'Score {k}' for k in scores],
        var_name='Score ID', value_name='Score')

    # Plot
    f, ax = plt.subplots(1, 1, figsize=tuple(figsize))
    sns.boxplot(
        x='Participant', y='Score', hue='Score ID', data=df,
        palette='muted')
    if datapoints:
        sns.swarmplot(
            x='Participant', y='Score', hue='Score ID', data=df,
            size=3, color='black', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:len(scores)], labels=labels[:len(scores)])

    return f, ax


#%% Main
if __name__ == '__main__':
    csv = r'/Volumes/NeuroTin-EEG/Data/scores_logs.csv'
    participant = 61
    plot_score_evolution_per_participant(csv, participant, scores=(1, 5, 10))
    plot_score_across_participants(csv, scores=10, datapoints=True)
