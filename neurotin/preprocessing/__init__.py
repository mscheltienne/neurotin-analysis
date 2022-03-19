"""Preprocessing module."""

from .bad_channels import (RANSAC_bads_suggestion,  # noqa: F401
                           PREP_bads_suggestion)
from .events import (add_annotations_from_events, check_events,  # noqa: F401
                     replace_event_value)
from .filters import apply_filter_aux, apply_filter_eeg  # noqa: F401
from .preprocessing import (prepare_raw, remove_artifact_ic,  # noqa: F401
                            pipeline)
