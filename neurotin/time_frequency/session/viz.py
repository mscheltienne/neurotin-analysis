import seaborn as sns
from matplotlib import pyplot as plt

from ...utils.docs import fill_doc
from ...utils.checks import _check_type, _check_participant, _check_participants


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
def diff_catplot_distribution(df_positives, df_negatives, participants,
                              **kwargs):
    """
    Plot the distribution of positive vs negative diff.

    Parameters
    ----------
    %(count_positives)s
    %(count_negatives)s
    participants : list
        List of the participant IDs to compare.

    Returns
    -------
    g : FacetGrid
    """
    participants = sorted(_check_participants(participants))

    # filter df
    df_positives = df_positives[df_positives['participant'].isin(participants)]
    df_negatives = df_negatives[df_negatives['participant'].isin(participants)]

    # create facetgrid with first catplot
    g = sns.catplot(x='session', y='count', row='participant',
                    row_order=participants, data=df_positives,
                    kind='bar', legend_out=True, sharey=True,
                    color='limegreen', **kwargs)

    # add second plot on each axes
    for k, ax in enumerate(g.axes):
        df = df_negatives[df_negatives['participant'] == participants[k]]
        sns.barplot(x='session', y='count', data=df,
                    color='deepskyblue', ax=ax[0])

    # add line plot 0.5 below positive value
    df = df_positives.copy()
    df.loc[:, 'count'] -= 0.5
    df.loc[:, 'session'] -= 1
    for k, ax in enumerate(g.axes):
        df_ = df[df['participant'] == participants[k]]
        sns.lineplot(x='session', y='count', data=df_, ax=ax[0], legend=False)

    # style
    for k, ax in enumerate(g.axes):
        ax[0].yaxis.set_visible(False)
        ax[0].axhline(y=0, linestyle='--', color='black', linewidth=0.5)

    return g
