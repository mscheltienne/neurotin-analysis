from typing import List, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mne.channels import make_standard_montage
from mne.io import BaseRaw
from mne.preprocessing import compute_bridged_electrodes as compute_bridged_electrodes_mne
from mne.viz import plot_bridged_electrodes as plot_bridged_electrodes_mne
from numpy.typing import NDArray

from ..utils._checks import _check_type


def plot_bridged_electrodes(
    raw: BaseRaw,
) -> Tuple[plt.Figure, NDArray[plt.Axes]]:
    """Compute and plot bridged electrodes.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance before filtering. The raw instance is copied, the EEG
        channels are picked and filtered between 0.5 and 30 Hz.

    Returns
    -------
    fig : Figure
    ax : Array of Axes
    """
    _check_raw(raw)

    # retrieve bridge electrodes, operates on a copy
    bridged_idx, ed_matrix = compute_bridged_electrodes_mne(raw)

    # create figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # plot electrical distances on the first row
    ed_plot = np.zeros(ed_matrix.shape[1:]) * np.nan
    triu_idx = np.triu_indices(ed_plot.shape[0], 1)
    for idx0, idx1 in np.array(triu_idx).T:
        ed_plot[idx0, idx1] = np.nanmedian(ed_matrix[:, idx0, idx1])

    im1 = ax[0, 0].imshow(ed_plot, aspect="auto")
    cax1 = fig.colorbar(im1, ax=ax[0, 0])
    cax1.set_label(r"Electrical Distance ($\mu$$V^2$)")
    ax[0, 0].set_xlabel("Channel Index")
    ax[0, 0].set_ylabel("Channel Index")

    im2 = ax[0, 1].imshow(ed_plot, aspect="auto", vmax=5)
    cax2 = fig.colorbar(im2, ax=ax[0, 1])
    cax2.set_label(r"Electrical Distance ($\mu$$V^2$)")
    ax[0, 1].set_xlabel("Channel Index")
    ax[0, 1].set_ylabel("Channel Index")

    # plot distribution
    ax[1, 0].hist(
        ed_matrix[~np.isnan(ed_matrix)], bins=np.linspace(0, 500, 51)
    )
    ax[1, 0].set_xlabel(r"Electrical Distance ($\mu$$V^2$)")
    ax[1, 0].set_ylabel("Count (channel pairs for all epochs)")
    ax[1, 0].set_title("Electrical Distance Matrix Distribution")

    # plot topographic map
    plot_bridged_electrodes_mne(
        raw.info.copy().set_montage("standard_1020"),
        bridged_idx,
        ed_matrix,
        title="Bridged Electrodes",
        topomap_args=dict(vmax=5, axes=ax[1, 1]),
    )

    fig.tight_layout()
    return fig, ax


def compute_bridged_electrodes(raw: BaseRaw) -> List[str]:
    """Compute the bridged electrodes.

    This function returns the list of channels to be excluded because of a
    gel-bridge between electrodes. It retains one electrode / bridge.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance before filtering. The raw instance is copied, the EEG
        channels are picked and filtered between 0.5 and 30 Hz.

    Returns
    -------
    bads : list of str
        List of channels to exclude because of a gel-bridge.
    """
    _check_raw(raw)

    # retrieve bridge electrodes, operates on a copy
    bridged_idx, ed_matrix = compute_bridged_electrodes_mne(raw)
    groups_idx = _find_groups(bridged_idx)
    groups = [[raw.ch_names[k] for k in group] for group in groups_idx]

    # retrieve montage
    positions = make_standard_montage("standard_1020")._get_ch_pos()

    bads = list()
    for group in groups:
        # remove the electrode(s) closest to the reference as they have more
        # adjacent electrodes to use for spatial interpolation.
        distances = [np.linalg.norm(positions["CPz"]-positions[ch]) for ch in group]
        idx = np.argmax(distances)
        bads.extend([ch for k, ch in enumerate(group) if k != idx])

    return bads


def _check_raw(raw: BaseRaw):
    """Check that the raw instance filters are compatible."""
    _check_type(raw, (BaseRaw,), "raw")
    if 0.5 < raw.info["highpass"]:
        raise RuntimeError(
            "The raw instance should not be highpass-filtered " "above 0.5 Hz."
        )
    if raw.info["lowpass"] < 30:
        raise RuntimeError(
            "The raw instance should not be lowpass-filtered " "below 30 Hz."
        )


def _find_groups(bridged_idx):
    """Find the groups of electrodes that are bridged."""
    G = nx.Graph()
    for bridge in bridged_idx:
        G.add_edge(*bridge)
    return [tuple(elt) for elt in nx.connected_components(G)]
