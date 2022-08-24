import itertools
from typing import Optional, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mne import create_info
from mne.io import BaseRaw, RawArray
from mne.preprocessing import (
    compute_bridged_electrodes as compute_bridged_electrodes_mne,
)
from mne.transforms import _cart_to_sph, _sph_to_cart
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


def interpolate_bridged_electrodes(
    raw: BaseRaw,
    limit: Optional[int] = 5,
    total_limit: Optional[int] = 16,
) -> BaseRaw:
    """Compute the bridged electrodes.

    This function returns the list of channels to be excluded because of a
    gel-bridge between electrodes. It retains one electrode / bridge.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance before filtering. The raw instance is copied, the EEG
        channels are picked and filtered between 0.5 and 30 Hz.
    limit : int | None
        Maximum number of electrodes (inc.) that can be bridged in a group
        before raising. If None, disables the limit.
    total_limit : int | None
        Maximum number of electrodes (inc.) that can be bridged before raising.
        If None, disables the limit.

    Returns
    -------
    raw : Raw
        MNE Raw instance before filtering where the bridged channels have been
        interpolated.
    """
    _check_raw(raw)
    if limit is not None:
        _check_type(limit, ("int",), "limit")
        assert 0 < limit
    if total_limit is not None:
        _check_type(total_limit, ("int",), "total_limit")
        assert 0 < total_limit

    # retrieve bridge electrodes, operates on a copy
    bridged_idx, ed_matrix = compute_bridged_electrodes_mne(raw)
    if total_limit is not None and total_limit <= len(
        set(itertools.chain(*bridged_idx))
    ):
        raise RuntimeError(
            f"More than {total_limit} electrodes have gel-bridges."
        )

    # find groups of electrodes
    G = nx.Graph()
    for bridge in bridged_idx:
        G.add_edge(*bridge)
    groups_idx = [tuple(elt) for elt in nx.connected_components(G)]
    if any(len(group) >= limit for group in groups_idx):
        raise RuntimeError(
            f"More than {limit} electrodes have a common gel-bridge."
        )

    # make virtual channels
    pos = raw.get_montage().get_positions()
    ch_pos = pos["ch_pos"]
    virtual_chs = dict()
    bads = set()
    data = raw.get_data()
    for k, group_idx in enumerate(groups_idx):
        group_names = [raw.ch_names[k] for k in group_idx]
        bads = bads.union(group_names)

        # compute midway position in spherical coordinates in "head"
        # (more accurate than cutting though the scalp by using cartesian)
        sphere_positions = np.zeros((len(group_idx), 3))
        for i, ch_name in enumerate(group_names):
            sphere_positions[i, :] = _cart_to_sph(ch_pos[ch_name])
        pos_virtual = _sph_to_cart(np.average(sphere_positions, axis=0))

        # create the virtual channel info and set the position
        virtual_info = create_info(
            ch_names=[f"virtual {k+1}"],
            sfreq=raw.info["sfreq"],
            ch_types="eeg",
        )
        virtual_info["chs"][0]["loc"][:3] = pos_virtual

        # create the virtual channel data array
        group_data = np.zeros((len(group_idx), data.shape[1]))
        for i, ch_idx in enumerate(group_idx):
            group_data[i, :] = data[ch_idx, :]
        virtual_data = np.average(group_data, axis=0).reshape(1, -1)

        # create the virtual channel
        virtual_ch = RawArray(virtual_data, virtual_info, raw.first_samp)
        virtual_chs[f"virtual {k+1}"] = virtual_ch

    # add the virtual channels
    raw.add_channels(list(virtual_chs.values()), force_update_info=True)

    # interpolate
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, mode="accurate")

    # drop virtual channels
    raw.drop_channels(list(virtual_chs.keys()))

    return raw


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
    assert raw.get_montage() is not None
