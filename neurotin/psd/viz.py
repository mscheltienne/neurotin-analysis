from typing import Union

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns

from ..utils.align_axes import align_yaxis
from ..utils._checks import _check_participants
from ..utils._docs import fill_doc


# -----------------------------------------------------------------------------
# Plots at the subject-level, representing the evolution of a metric across the
# 15 session. Multiple subjects can be compared at once.
# x-axis: session ids / y-axis: metric
# -----------------------------------------------------------------------------
@fill_doc
def catplot(
        df,
        participants: Union[int, list, tuple],
        kind: str = 'box',
        **kwargs
        ):
    """
    Plot a comparison between participants.

    Parameters
    ----------
    %(df_psd)s
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


def captlot_success_distribution(
        df_positives,
        df_negatives,
        participants: Union[int, list, tuple],
        **kwargs
        ):
    """
    Plot the distribution of positive vs negative diff.

    Parameters
    ----------
    df_positives : DataFrame
        Counts of positives 'diff'.
    df_negatives : DataFrame
        Counts of negatives 'diff'.
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


# -----------------------------------------------------------------------------
# Plots at the session-level, all or some sessions have been merged together.
# Multiple subjects can be compared at once.
# x-axis: participant ids / y-axis: metric
# -----------------------------------------------------------------------------
def plot_joint_clinical_nfb_performance(
        df,
        df_clinical,
        name: str,
        participants: Union[int, list, tuple]
        ):
    """
    Plot a joint figure with both clinical outcomes and NFB performance.

    Parameters
    ----------
    df_positives : DataFrame
        Counts of positives 'diff'. All sessions are merged and a single count
        per session is stored.
    clinical : DtaFrame
        Clinical dataframe to plot below the success rate.
    name : str
        Clinical result name.
    participants : list | tuple
        List of participants ID to include.
    """
    participants = sorted(_check_participants(participants))

    # filter df
    df = df[df['participant'].isin(participants)]
    # inverse value for df_clinical
    df_clinical['result'] = -df_clinical['result']

    # create figure
    f, ax1 = plt.subplots(1, 1, figsize=(15, 5))

    # define order by performance
    order = [(idx, score) for idx, score in df.values]
    order = sorted(order, key=lambda x: x[1])
    order = [int(elt[0]) for elt in order]

    # create bar plot for nfb performance
    sns.barplot(x='participant', y='count', data=df, color='slategrey',
                order=order, ax=ax1)

    # figure out hue_order
    orders = []
    for participant in df_clinical['participant'].unique():
        df_ = df_clinical[df_clinical['participant'] == participant].dropna()
        ordered = sorted(
            [(visit, date) for visit, date in df_[['visit', 'date']].values],
            key=lambda x: x[1])
        orders.append([elt[0] for elt in ordered])
    # Take the longest
    hue_order = sorted(orders, key=lambda x: len(x))[-1]

    # create bar plot for clinical results
    ax2 = ax1.twinx()
    sns.barplot(data=df_clinical, x='participant', y='result', hue='visit',
                order=order, hue_order=hue_order, ax=ax2, palette='deep')

    # set y-ticks
    miny, maxy = ax2.get_ylim()
    yticks = np.arange(-(abs(miny) + 10 - abs(miny) % 10), 0, 10)
    ax2.set_yticks(yticks)
    ax2.set_ylabel(f'{name} scores (lower is better)')
    ax2.set_yticklabels([str(-k) for k in yticks])
    ax1.set_yticks(np.arange(0, 1, 0.2))
    ax1.set_ylabel('Percentage of NFB successful blocks')
    ax1.yaxis.set_major_formatter(PercentFormatter(1))

    # align y-axis
    align_yaxis(ax1, ax2)

    # set legend
    ax2.legend(title='Visit', loc='center left', bbox_to_anchor=(1.1, 0.5))
    f.subplots_adjust(left=0.08, right=0.75)

    # add horizontal line at 50%
    ax1.axhline(0.5, linestyle='--', linewidth=1, color='black')


# -----------------------------------------------------------------------------
# Plots at the group-level, all sessions and all participants have been merged
# together.
# -----------------------------------------------------------------------------
