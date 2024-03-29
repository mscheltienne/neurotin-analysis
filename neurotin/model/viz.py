from typing import Optional, Union

import mne
import numpy as np
import pandas as pd
from mne.io import Info
from numpy.typing import NDArray

from ..utils._checks import _check_type


def plot_topomap(
    weights: Union[pd.Series, NDArray[float]],
    info: Optional[Info] = None,
    **kwargs,
) -> None:
    """Plot the weights as a topographic map.

    Parameters
    ----------
    weights : array | Series
        If a numpy array is provided, the channel names must be provided in
        info. If a Series is provided, the channel names are retrieved from
        the index.
    info : Info
       MNE measurement information instance containing the channel names.
       A standard 1020 montage is added.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    weights = _check_type(
        weights, (pd.Series, np.ndarray), item_name="weights"
    )
    info = _check_type(info, (None, mne.io.Info), item_name="info")

    if isinstance(weights, pd.Series):
        data = weights.values
        info = mne.create_info(list(weights.index), 1, "eeg")
        info.set_montage("standard_1020")
    else:
        _check_info(info, weights.size)
        info.set_montage("standard_1020")

    im, cn = mne.viz.plot_topomap(data, pos=info, **kwargs)
    return im, cn


def _check_info(info: Optional[Info], n: int) -> None:
    """Check that info is valid if weights is a numpy array."""
    if info is None:
        raise ValueError("Info must be provided if weights is a numpy array.")
    assert (
        len(info.ch_names) == n
    ), "Info does not contain the same number of channels as weights."
