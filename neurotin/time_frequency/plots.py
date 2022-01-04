import seaborn as sns
from matplotlib import pyplot as plt

from ..utils.docs import fill_doc
from ..utils.checks import _check_type, _check_participant, _check_participants


@fill_doc
def violinplot(df, participant, figsize=(10, 5), **kwargs):
    """
    Plot as violin plots (left: non-regulation; right: regulation) the average
    PSD in a given band for a given participant.

    Parameters
    ----------
    %(psd_df)s
    %(participant)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_participant(participant)
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.violinplot(x='session', y='avg', hue='phase', palette='muted',
                   hue_order=['non-regulation', 'regulation'],
                   data=df[df['participant'] == participant], split=True,
                   scale="count", inner="quartile", ax=ax, **kwargs)

    return f, ax


@fill_doc
def boxplot(df, participant, figsize=(10, 5), **kwargs):
    """
    Plot as box plots (left: non-regulation; right: regulation) the average
    PSD in a given band for a given participant.

    Parameters
    ----------
    %(psd_df)s
    %(participant)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_participant(participant)
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(x='session', y='avg', hue='phase', palette='muted',
                hue_order=['non-regulation', 'regulation'],
                data=df[df['participant'] == participant], ax=ax, **kwargs)

    return f, ax


@fill_doc
def catplot(df, participants, kind='box', **kwargs):
    """
    Plot a comparison between participants.

    Parameters
    ----------
    %(psd_df)s
    participants : list
        List of the participant IDs to compare.
    kind : str
        The kind of plot to draw, corresponds to the name of a categorical
        axes-level plotting function. Options are: “strip”, “swarm”, “box”,
        “violin”, “boxen”, “point”, “bar”, or “count”.

    Returns
    -------
    g : FacetGrid
    """
    participants = _check_participants(participants)
    g = sns.catplot(x='session', y='avg', row='participant',
                    row_order=sorted(participants), hue='phase',
                    hue_order=['non-regulation', 'regulation'],
                    palette='muted',
                    data=df[df['participant'].isin(participants)],
                    kind=kind, legend_out=True, sharey=False, **kwargs)

    return g


@fill_doc
def diff_lineplot(df, participant, figsize=(10, 5), **kwargs):
    """
    Plot the difference between PSD power in consecutive non-regulation and
    regulation phase.

    Parameters
    ----------
    %(psd_diff_df)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_participant(participant)
    _check_type(figsize, (tuple, ), item_name='figsize')

    # extract information
    df = df[df['participant'] == participant]
    df.sort_values(by=['session', 'run', 'idx'], ascending=True)
    df.reset_index()

    # create figure
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(x=df.index, y='diff', data=df, ax=ax)

    return f, ax
