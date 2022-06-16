"""Preprocessing module."""

from .bridge import plot_bridged_electrodes  # noqa: F401
from .preprocessing import (  # noqa: F401
    pipeline,
    prepare_raw,
    preprocess,
    remove_artifact_ic,
)
