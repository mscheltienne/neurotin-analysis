"""Fill docstrings to avoid redundant docstrings in multiple files.

Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""

import sys
from typing import Callable, Dict, List

# ------------------------- Documentation dictionary -------------------------
docdict = dict()

# ---------------------------------- verbose ---------------------------------
docdict[
    "verbose"
] = """
verbose : int | str | bool | None
    Sets the verbosity level. The verbosity increases gradually between
    "CRITICAL", "ERROR", "WARNING", "INFO" and "DEBUG".
    If None is provided, the verbosity is set to "WARNING".
    If a bool is provided, the verbosity is set to "WARNING" for False and to
    "INFO" for True."""

# ------------------------------ MNE objects ---------------------------------
docdict[
    "raw"
] = """
raw : Raw
    MNE raw object (continuous data)."""
docdict[
    "ica"
] = """
ica : ICA
    Fitted ICA decomposition using the Preconditioned ICA for Real Data
    algorithm (PICARD)."""

# -------------------------------- general -----------------------------------
docdict[
    "folder_raw_data"
] = """
folder : path-like
    Path to the directory containing raw data with recordings, logs and
    models."""
docdict[
    "folder_pp_data"
] = """
folder_pp : path-like
    Path to the directory containing preprocessed data."""
docdict[
    "participant"
] = """
participants : int
    ID of the participant to include."""
docdict[
    "participants"
] = """
participants : list | tuple
    List of participant IDx to include."""
docdict[
    "session"
] = """
session : int
    ID of the session to include (between 1 and 15)."""
docdict[
    "copy"
] = """
copy : bool
    If True, operates on a copy and return a copy."""
docdict[
    "valid_only"
] = """
valid_only : bool
    If True, return only the valid runs."""
docdict[
    "regular_only"
] = """
regular_only : bool
    If True, return only regular neurofeedback runs."""
docdict[
    "transfer_only"
] = """
transfer_only : bool
    If True, returns only transfer neurofeedback runs."""
docdict[
    "n_jobs"
] = """
n_jobs : int
    Number of parallel jobs used. Must not exceed the core count. Can be -1
    to use all cores."""


# --------------------------------- evamed -----------------------------------
docdict[
    "df_raw_evamed"
] = """
df : DataFrame
    DataFrame loaded by neurotin.io.read_csv_evamed()."""
docdict[
    "df_clinical"
] = """
df: DataFrame
    DataFrame containing the columns 'participant, 'visit', 'date', and
    'result'."""

# ------------------------------- band-power ---------------------------------
docdict[
    "df_bp"
] = """
df : DataFrame
    Band-power between [fmin, fmax]. The columns are:
        participant : int - Participant ID
        session : int - Session ID (1 to 15)
        run : int - Run ID
        phase : str - 'regulation' or 'non-regulation'
        idx : ID of the phase within the run (0 to 9)
        ch : float - Averaged PSD (1 per channel)"""
docdict[
    "bp_duration"
] = """
duration : float
    Duration of an epoch in seconds."""
docdict[
    "bp_overlap"
] = """
overlap :float
    Duration of epoch overlap in seconds."""

# -------------------------------- matplotlib --------------------------------
docdict[
    "plt.figsize"
] = """
figsize : tuple
    2-sequence tuple defining the matplotlib figure size: (width, height) in
    inches."""

# ------------------------- Documentation functions --------------------------
docdict_indented: Dict[int, Dict[str, str]] = dict()


def fill_doc(f: Callable) -> Callable:
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).

    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = " " * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [
                indent + line if k != 0 else line
                for k, line in enumerate(docstr.strip().splitlines())
            ]
            indented[name] = "\n".join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")

    return f


def _indentcount_lines(lines: List[str]) -> int:
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indent = sys.maxsize
    for k, line in enumerate(lines):
        if k == 0:
            continue
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    return indent


def copy_doc(source: Callable) -> Callable:
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : callable
        The function to copy the docstring from.

    Returns
    -------
    wrapper : callable
        The decorated function.

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise RuntimeError(
                f"The docstring from {source.__name__} could not be copied "
                "because it was empty."
            )
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper
