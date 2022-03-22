from matplotlib import pyplot as plt
import pandas as pd
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
    valids = ('Baseline', 'Pre-assessment', 'Post-assessment')
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

    order = sorted(df.visit.unique())
    sns.boxplot(x='visit', y='result', data=df, order=order, ax=ax)
    ax.set(xlabel="Visit", ylabel=f"{name} Score (lower is better)")

    return f, ax


def barplot_evolution(df, name, figsize=(10, 5)):
    """Plot the difference Baseline - Post-assessment for each participant
    as a bar plot.

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

    # extract difference between post-assessment and baseline
    order = ('Baseline', 'Post-assessment')
    df = df[df['visit'].isin(order)]

    df_dict = {'participant': [], 'result':[]}
    for participant in df['participant'].unique():
        df_ = df[df['participant'] == participant]
        post = df_[df_['visit'] == order[1]]['result'].values[0]
        base = df_[df_['visit'] == order[0]]['result'].values[0]
        result = post - base

        # fill dict
        df_dict['participant'].append(participant)
        df_dict['result'].append(result)

    df = pd.DataFrame.from_dict(df_dict, orient='columns')

    # order to plot
    order = [(idx, score) for idx, score in df.values]
    order = sorted(order, key=lambda x: x[1])
    order = [int(elt[0]) for elt in order]

    # plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(x='participant', y='result', data=df, color="steelblue",
                order=order, ax=ax)

    # set labels
    ax.set_xlabel('Participant ID')
    ax.set_ylabel(f'{name} Post - Baseline')

    return f, ax
