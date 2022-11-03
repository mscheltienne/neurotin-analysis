from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mne.time_frequency import read_tfrs

from ..utils._checks import (
    _check_participant,
    _check_path,
    _check_type,
    _check_value,
)
from .tfr import METHODS


def plot_tfr_subject(
    folder_tfr: Union[str, Path],
    participant: int,
    method: str,
    regular_only: bool = False,
    transfer_only: bool = False,
    timefreqs: Optional[
        Union[
            List[Tuple[float, float]],
            Dict[Tuple[float, float], Tuple[float, float]],
        ]
    ] = None,
    resolutions: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot a subject-level TFR."""
    folder_tfr = _check_path(
        folder_tfr, item_name="folder_tfr", must_exist=True
    )
    participant = _check_participant(participant)
    _check_type(method, (str,), "method")
    _check_value(method, METHODS, "method")
    _check_type(regular_only, (bool,), "regular_only")
    _check_type(transfer_only, (bool,), "transfer_only")
    assert (
        (regular_only or transfer_only)
        and not (regular_only and transfer_only)
    ) or (not regular_only and not transfer_only)
    _check_type(resolutions, (None, tuple), "resolutions")
    if resolutions is not None:
        assert len(resolutions) == 2
        assert all(0 < r for r in resolutions)

    # figure out where to look for the tfr
    if regular_only:
        folder = folder_tfr / method / "regular"
    elif transfer_only:
        folder = folder_tfr / method / "transfer"
    else:
        folder = folder_tfr / method / "full"
    fname = folder / f"sub-{participant}-tfr.h5"

    # load and baseline
    tfr = read_tfrs(fname)
    assert len(tfr) == 1
    tfr = tfr[0]
    tfr.apply_baseline((2, 6), mode="percent")

    # plot
    if timefreqs is None:
        fig = tfr.plot(combine="mean")[0]
    else:
        fig = tfr.plot_joint(timefreqs, combine="mean")

    fig.axes[0].axvline(x=8, color="darkslategray", linestyle="--")
    fig.axes[0].text(
        0.160,
        1.03,
        "Rest",
        horizontalalignment="center",
        verticalalignment="center",
        transform=fig.axes[0].transAxes,
    )
    fig.axes[0].text(
        0.69,
        1.03,
        "Regulation",
        horizontalalignment="center",
        verticalalignment="center",
        transform=fig.axes[0].transAxes,
    )
    if resolutions is not None:  # resolutions is (time, frequency)
        rect = Rectangle(
            (1, 15 - 1 - resolutions[1]),
            resolutions[0],
            resolutions[1],
            linewidth=1,
            edgecolor="darkslategray",
            facecolor="none",
        )
        fig.axes[0].add_patch(rect)
    fig.suptitle(f"Subject {participant}")

    return fig


def plot_itc_subject(
    folder_tfr: Union[str, Path],
    participant: int,
    method: str,
    regular_only: bool = False,
    transfer_only: bool = False,
    timefreqs: Optional[
        Union[
            List[Tuple[float, float]],
            Dict[Tuple[float, float], Tuple[float, float]],
        ]
    ] = None,
    resolutions: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot a subject-level ITC."""
    folder_tfr = _check_path(
        folder_tfr, item_name="folder_tfr", must_exist=True
    )
    participant = _check_participant(participant)
    _check_type(method, (str,), "method")
    _check_value(method, METHODS, "method")
    _check_type(regular_only, (bool,), "regular_only")
    _check_type(transfer_only, (bool,), "transfer_only")
    assert (regular_only or transfer_only) or (
        not regular_only and not transfer_only
    )
    if resolutions is not None:
        assert len(resolutions) == 2
        assert all(0 < r for r in resolutions)

    # figure out where to look for the tfr
    if regular_only:
        folder = folder_tfr / method / "regular"
    elif transfer_only:
        folder = folder_tfr / method / "transfer"
    else:
        folder = folder_tfr / method / "full"
    fname = folder / f"sub-{participant}-itc.h5"

    # load and apply baseline
    itc = read_tfrs(fname)
    assert len(itc) == 1
    itc = itc[0]
    itc.apply_baseline((2, 6), mode="percent")

    # plot
    if timefreqs is None:
        fig = itc.plot(combine="mean")[0]
    else:
        fig = itc.plot_joint(timefreqs, combine="mean")

    fig.axes[0].axvline(x=8, color="darkslategray", linestyle="--")
    fig.axes[0].text(
        0.160,
        1.03,
        "Rest",
        horizontalalignment="center",
        verticalalignment="center",
        transform=fig.axes[0].transAxes,
    )
    fig.axes[0].text(
        0.69,
        1.03,
        "Regulation",
        horizontalalignment="center",
        verticalalignment="center",
        transform=fig.axes[0].transAxes,
    )
    if resolutions is not None:  # resolutions is (time, frequency)
        rect = Rectangle(
            (1, 15 - 1 - resolutions[1]),
            resolutions[0],
            resolutions[1],
            linewidth=1,
            edgecolor="darkslategray",
            facecolor="none",
        )
        fig.axes[0].add_patch(rect)
    fig.suptitle(f"Subject {participant}")

    return fig


def plot_tfr_session(
    folder_tfr: Union[str, Path],
    participant: int,
    method: str,
    groupby: int = 1,
    figsize: Tuple[float, float] = None,
) -> plt.Figure:
    """Plot a session-level TFR."""
    folder_tfr = _check_path(
        folder_tfr, item_name="folder_tfr", must_exist=True
    )
    participant = _check_participant(participant)
    _check_type(method, (str,), "method")
    _check_value(method, METHODS, "method")
    _check_type(groupby, ("int",), "groupby")
    _check_value(groupby, (1, 3, 5), "groupby")
    _check_type(resolutions, (None, tuple), "resolutions")

    # figure out where to look for the tfr and create figure
    if groupby == 1:
        folder = folder_tfr / method / "session-level"
        figsize = (20, 10) if figsize is None else figsize
        fig, ax = plt.subplots(3, 5, figsize=figsize)
    elif groupby == 3:
        folder = folder_tfr / method / "session-level-groupby-3"
        figsize = (20, 5) if figsize is None else figsize
        fig, ax = plt.subplots(1, 5, figsize=figsize)
    elif groupby == 5:
        folder = folder_tfr / method / "session-level-groupby-5"
        figsize = (20, 5) if figsize is None else figsize
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    # list files to plot
    files = [
        file
        for file in folder.iterdir()
        if f"sub-{participant}" in file.stem and file.stem.endswith("-tfr")
    ]
    files = sorted(files, key=lambda x: x.stem.split("ses-")[1])

    mapping = {
        "1-2-3": 0,
        "4-5-6": 1,
        "7-8-9": 2,
        "10-11-12": 3,
        "13-14-15": 4,
        "1-2-3-4-5": 0,
        "6-7-8-9-10": 1,
        "11-12-13-14-15": 2,
    }
    axes_used = list()
    for file in files:
        # figure out on which axes to plot
        sessions = file.stem.split("ses-")[1].split("-tfr")[0]
        if groupby == 1:
            assert sessions not in mapping
            k = int(sessions) - 1
            axes = ax[k // 5, k % 5]
        else:
            assert sessions in mapping
            k = mapping[sessions]
            axes = ax[k]
        axes_used.append(k)

        # load tfr and apply baseline
        tfr = read_tfrs(file)
        assert len(tfr) == 1
        tfr = tfr[0]
        tfr.apply_baseline((2, 6), mode="percent")
        # plot
        tfr.plot(combine="mean", axes=axes)

    # format figure
    for k, a in enumerate(ax.flatten()):
        if groupby == 1:
            a.set_title(f"S{k+1}")
        elif groupby == 3:
            a.set_title(f"S{tuple(range(3*k+1, 3*(k+1)+1))}")
        elif groupby == 5:
            a.set_title(f"S{tuple(range(5*k+1, 5*(k+1)+1))}")

        if k not in axes_used:
            # hide the axes and write "No Data"
            a.text(
                0.5,
                0.5,
                "No Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=a.transAxes,
            )
            a.axis("off")

    fig.tight_layout()
    return fig


def plot_itc_session(
    folder_tfr: Union[str, Path],
    participant: int,
    method: str,
    groupby: int = 1,
    figsize: Tuple[float, float] = None,
) -> plt.Figure:
    """Plot a session-level ITC."""
    folder_tfr = _check_path(
        folder_tfr, item_name="folder_tfr", must_exist=True
    )
    participant = _check_participant(participant)
    _check_type(method, (str,), "method")
    _check_value(method, METHODS, "method")
    _check_type(groupby, ("int",), "groupby")
    _check_value(groupby, (1, 3, 5), "groupby")

    # figure out where to look for the tfr and create figure
    if groupby == 1:
        folder = folder_tfr / method / "session-level"
        figsize = (20, 10) if figsize is None else figsize
        fig, ax = plt.subplots(3, 5, figsize=figsize)
    elif groupby == 3:
        folder = folder_tfr / method / "session-level-groupby-3"
        figsize = (20, 5) if figsize is None else figsize
        fig, ax = plt.subplots(1, 5, figsize=figsize)
    elif groupby == 5:
        folder = folder_tfr / method / "session-level-groupby-5"
        figsize = (20, 5) if figsize is None else figsize
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    # list files to plot
    files = [
        file
        for file in folder.iterdir()
        if f"sub-{participant}" in file.stem and file.stem.endswith("-itc")
    ]
    files = sorted(files, key=lambda x: x.stem.split("ses-")[1])

    mapping = {
        "1-2-3": 0,
        "4-5-6": 1,
        "7-8-9": 2,
        "10-11-12": 3,
        "13-14-15": 4,
        "1-2-3-4-5": 0,
        "6-7-8-9-10": 1,
        "11-12-13-14-15": 2,
    }
    axes_used = list()
    for file in files:
        # figure out on which axes to plot
        sessions = file.stem.split("ses-")[1].split("-itc")[0]
        if groupby == 1:
            assert sessions not in mapping
            k = int(sessions) - 1
            axes = ax[k // 5, k % 5]
        else:
            assert sessions in mapping
            k = mapping[sessions]
            axes = ax[k]
        axes_used.append(k)

        # load tfr and apply baseline
        itc = read_tfrs(file)
        assert len(itc) == 1
        itc = itc[0]
        itc.apply_baseline((2, 6), mode="percent")
        # plot
        itc.plot(combine="mean", axes=axes)

    # format figure
    for k, a in enumerate(ax.flatten()):
        if groupby == 1:
            a.set_title(f"S{k+1}")
        elif groupby == 3:
            a.set_title(f"S{tuple(range(3*k+1, 3*(k+1)+1))}")
        elif groupby == 5:
            a.set_title(f"S{tuple(range(5*k+1, 5*(k+1)+1))}")

        if k not in axes_used:
            # hide the axes and write "No Data"
            a.text(
                0.5,
                0.5,
                "No Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=a.transAxes,
            )
            a.axis("off")

    fig.tight_layout()
    return fig
