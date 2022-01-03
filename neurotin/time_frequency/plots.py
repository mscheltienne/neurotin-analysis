import seaborn as sns
from matplotlib import pyplot as plt

from ..utils.checks import _check_type, _check_participant, _check_participants


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


def violinplot_comparison(df, participants, figsize=None, **kwargs):
    """
    Plot a comparison between participants.

    Parameters
    ----------
    %(psd_df)s
    participants : list
        List of the participant IDs to compare.

    Returns
    -------
    f : Figure
    ax : Axes
    """
    participants = _check_participants(participants)
    _check_type(figsize, (tuple, None), item_name='figsize')
    figsize = (20, 2*len(participants)) if figsize is None else figsize

    f, ax = plt.subplots(len(participants), 1, figsize=figsize)
    for k, participant in enumerate(participants):
        sns.violinplot(x='session', y='avg', hue='phase', palette='muted',
                       hue_order=['non-regulation', 'regulation'],
                       data=df[df['participant'] == participant], split=True,
                       scale="count", inner="quartile", ax=ax[k], **kwargs)

    # design
    for ax_ in ax:
        ax_.legend().set_visible(False)
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    # adjust space
    f.subplots_adjust(hspace=0)

    # add session axis
    ax[-1].get_xaxis().set_visible(True)
    ax[-1].set_xticks(range(0, 15, 1))
    ax[-1].set_xticklabels(sorted(df['session'].unique()))
    ax[-1].set_xlabel('Session n°')

    return f, ax


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


def boxplot_comparison(df, participants, figsize=None, **kwargs):
    """
    Plot a comparison between participants.

    Parameters
    ----------
    %(psd_df)s
    participants : list
        List of the participant IDs to compare.

    Returns
    -------
    f : Figure
    ax : Axes
    """
    participants = _check_participants(participants)
    _check_type(figsize, (tuple, None), item_name='figsize')
    figsize = (20, 2*len(participants)) if figsize is None else figsize

    f, ax = plt.subplots(len(participants), 1, figsize=figsize)
    for k, participant in enumerate(participants):
        sns.boxplot(x='session', y='avg', hue='phase', palette='muted',
                    hue_order=['non-regulation', 'regulation'],
                    data=df[df['participant'] == participant], ax=ax[k],
                    **kwargs)

    # design
    for ax_ in ax:
        ax_.legend().set_visible(False)
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    # adjust space
    f.subplots_adjust(hspace=0)

    # add session axis
    ax[-1].get_xaxis().set_visible(True)
    ax[-1].set_xticks(range(0, 15, 1))
    ax[-1].set_xticklabels(sorted(df['session'].unique()))
    ax[-1].set_xlabel('Session n°')

    return f, ax
