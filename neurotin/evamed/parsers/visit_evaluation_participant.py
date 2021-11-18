"""Parser of visit evaluation (participant) Evamed questionnaires."""

import re

import pandas as pd

from ...utils.checks import _check_participant


def parse_visit_evaluation_participant(df, participant):
    """Parse dataframe and extract visit evaluation (participant) answers and
    information."""
    _check_participant(participant)
