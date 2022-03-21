from matplotlib import pyplot as plt
import seaborn as sns

from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def lineplot_evolution(df, name, figsize=(10, 5)):
    """Plot the evolution of clinical results from multiple participants and
    visits as lineplots with markers for the different visits.

    Parameters
    ----------
    %(df_clinical)s
    name : str
        Name of the questionnaire used.
    %(plt.figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set(xlabel="Date", ylabel=f"{name} Score")

    # baseline to post-assessment
    valids = ['Baseline', 'Pre-assessment', 'Post-assessment']
    location = df['visit'].isin(valids)
    sns.lineplot(data=df.loc[location], x='date', y='result',
                 hue='participant', palette='muted', markers=False,
                 dashes=False, legend=False, ax=ax)
    # late assessment
    valids = []  # TODO: add late assessment to parser
    location = df['visit'].isin(valids)
    dashes = [(2, 2)] * len(df['participant'].unique())
    sns.lineplot(data=df.loc[location], x='date', y='result',
                 hue='participant', palette='muted', markers=False,
                 dashes=dashes, legend=False, ax=ax)
    # markers
    sns.scatterplot(data=df, x='date', y='result', style='visit',
                    style_order=('Baseline', 'Pre-assessment',
                                 'Post-assessment'),
                    hue='participant', palette='muted', legend=True, s=50,
                    ax=ax)

    return f, ax


@fill_doc
def boxplot_visits(df, name, figsize=(5, 5)):
    """Plot the comparison between different visits for a clinical outcome as
    boxplots.

    Parameters
    ----------
    %(df_clinical)s
    name : str
        Name of the questionnaire used.
    %(plt.figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set(xlabel="Visit", ylabel=f"{name} Score (lower is better)")

    order = sorted(df.visit.unique())
    sns.boxplot(x='visit', y='result', data=df, order=order)

    return f, ax
