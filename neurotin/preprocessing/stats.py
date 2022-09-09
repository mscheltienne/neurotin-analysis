"""Stats about the preprocessing."""

from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..utils._checks import _check_path
from ..utils._docs import fill_doc
from ..utils.selection import list_runs


@fill_doc
def hist_dropouts(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    participants: Union[int, List[int], Tuple[int, ...]],
):
    """Plot the number of runs that have been dropped out during preprocessing.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(participants)s

    Returns
    -------
    f : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes of shape (1,).
    """
    folder_pp = _check_path(folder_pp, "folder", must_exist=True)
    runs = list_runs(folder, participants)

    # counters
    totals = {key: 0 for key in runs}
    dropped = {key: 0 for key in runs}
    for participant in runs:
        # flatten all sessions together
        files = list(chain(*runs[participant].values()))
        totals[participant] = len(files)

        for file in files:
            file_pp = folder_pp / Path(file).relative_to(folder)
            if not file_pp.exists():
                dropped[participant] += 1

    # convert to dataframes
    totals = pd.DataFrame.from_dict(totals, orient="index", columns=["total"])
    dropped = pd.DataFrame.from_dict(
        dropped, orient="index", columns=["dropped"]
    )

    f, ax = plt.subplots(1, 1)
    sns.barplot(x=totals.index, y=totals["total"], color='darkblue', ax=ax)
    sns.barplot(x=dropped.index, y=dropped["dropped"], color='lightblue', ax=ax)
    ax.set(xlabel="Participants", ylabel="Dropped / Total files")

    return f, ax
