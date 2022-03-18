from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns

from ...utils.docs import fill_doc
from ...utils.checks import _check_participants


@fill_doc
def plot_joint_clinical_nfb_performance(df, df_clinical, name, participants):
    """
    Plot a joint figure with both clinical outcomes and NFB performance.

    Parameters
    ----------
    %(count_positives)s
    clinical : DtaFrame
        Clinical dataframe to use.
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
    f, ax1 = plt.subplots(1, 1, figsize=(10, 3))

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
            [(when, date) for when, date in df_[['When', 'date']].values],
            key=lambda x: x[1])
        orders.append([elt[0] for elt in ordered])
    # Take the longest
    hue_order = sorted(orders, key=lambda x: len(x))[-1]

    # create bar plot for clinical results
    ax2 = ax1.twinx()
    sns.barplot(data=df_clinical, x='participant', y='result', hue='When',
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


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]
