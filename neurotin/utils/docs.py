"""
Fill function docstrings to avoid redundant docstrings in multiple files.
Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys


# ------------------------- Documentation dictionary -------------------------
docdict = dict()

# --------------------------------- general ----------------------------------
docdict['raw_in_place'] = """
raw : mne.io.Raw
    Raw instance (modified in-place)."""

# --------------------------------- pipeline ---------------------------------
docdict['pipeline_header'] = """Pipeline function called on each raw file."""
docdict['fname'] = """
fname : str | Path
    Path to the input '-raw.fif' file to process."""
docdict['success'] = """
success : bool
    False if a processing step raised an Exception."""

# ----------------------------------- cli ------------------------------------
docdict['cli_header'] = """CLI procesisng pipeline."""
docdict['input_dir_fif'] = """
input_dir_fif : str | Path
    Path to the folder containing the FIF files to process."""
docdict['output_dir_fif'] = """
output_dir_fif : str | Path
    Path to the folder containing the FIF files processed. The FIF files are
    saved under the same relative folder structure as in input_dir_fif."""
docdict['output_dir_fif_with_None'] = """
output_dir_fif : str | Path | None
    Path to the folder containing the FIF files processed. The FIF files are
    saved under the same relative folder structure as in input_dir_fif. If
    None, the input file is overwritten in-place."""
docdict['output_dir_set'] = """
output_dir_set : str | Path
    Path to the folder containing the SET files processed."""
docdict['n_jobs'] = """
n_jobs : int
    Number of parallel jobs used. Must not exceed the core count. Can be -1
    to use all cores."""
docdict['select_participant'] = """
participant : int | None
    Restricts file selection to this participant."""
docdict['select_session'] = """
session : int | None
    Restricts file selection to this session."""
docdict['select_fname'] = """
fname : str | Path | None
    Restrict file selection to this file (must be inside input_dir_fif)."""
docdict['ignore_existing'] = """
ignore_existing : bool
    If True, files already processed are not included."""

# --------------------------------- meas_info --------------------------------
docdict['raw_dir_fif'] = """
raw_dir_fif : str | Path
    Path to the directory containing raw data with logs files (used to set
    measurement date)."""
docdict['subject'] = """
subject : int
    ID of the subject."""
docdict['sex'] = """
sex : int
    Sex of the subject. 1: Male - 2: Female."""
docdict['birthday'] = """
birthday : 3-length tuple of int
    Subject's birthday as (year, month, day)."""
docdict['subject_info_fname'] = """
subject_info_fname : str | Path
    Path to the subject info file."""

# -------------------------------- prepare_raw -------------------------------
docdict['bads'] = """
bads : list
    List of selected and interpolated bad channels."""

# ------------------------- Documentation functions --------------------------
docdict_indented = dict()


def fill_doc(f):
    """
    Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = ' ' * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [indent+line if k != 0 else line
                     for k, line in enumerate(docstr.strip().splitlines())]
            indented[name] = '\n'.join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))

    return f


def _indentcount_lines(lines):
    """
    Minimum indent for all lines in line list.

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
    for line in lines:
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    if indent == sys.maxsize:
        return 0
    return indent


def copy_doc(source):
    """
    Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

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
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper