import mne


def plot_topomap(weights, info=None, **kwargs):
    """
    Plot the weights as a topographic map.

    Parameters
    ----------
    weights : array | DataFrame
        If a numpy array is provided, the channel names must be provided in
        info. If a DataFrame is provided, the channel names are retrieved from
        the index
    """
    mne.viz.plot_topomap(data, pos=info, **kwargs)
