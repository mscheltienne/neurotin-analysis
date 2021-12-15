import seaborn as sns
from matplotlib import pyplot as plt

from ..utils.checks import _check_type


def plot_multi_thi_evolution(df, figsize=(10, 5)):
    """Plot the THI results from multiple THI questionnaires/participants."""
    _check_type(figsize, (tuple, ), item_name='figsize')

    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(x='date', y='result', hue='participant', data=df,
                 palette='muted', style='participant', markers=True,
                 dashes=False, ax=ax)
    ax.set(xlabel="Date", ylabel="THI Score")

    return f, ax
