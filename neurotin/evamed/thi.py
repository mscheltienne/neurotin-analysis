import seaborn as sns
from matplotlib import pyplot as plt

from ..utils.checks import _check_type


def plot_multi_thi_evolution(df, figsize=(10, 5)):
    """Plot the THI results from multiple THI questionnaires/participants."""
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set(xlabel="Date", ylabel="THI Score")

    # baseline to post-assessment
    valids = ['Baseline', 'Pre-assessment', 'Post-assessment']
    location = df['When'].isin(valids)
    sns.lineplot(data=df.loc[location], x='date', y='result',
                 hue='participant', palette='muted', markers=False,
                 dashes=False, legend=False, ax=ax)
    # late assessment
    valids = []  # TODO: add late assessment to parser
    location = df['When'].isin(valids)
    dashes = [(2, 2)] * len(df['participant'].unique())
    sns.lineplot(data=df.loc[location], x='date', y='result',
                 hue='participant', palette='muted', markers=False,
                 dashes=dashes, legend=False, ax=ax)
    # markers
    sns.scatterplot(data=df, x='date', y='result', style='When',
                    style_order=('Baseline', 'Pre-assessment',
                                 'Post-assessment'),
                    hue='participant', palette='muted', legend=True, s=50,
                    ax=ax)

    return f, ax
