"""
Fill function docstrings to avoid redundant docstrings in multiple files.
Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys


# ------------------------- Documentation dictionary -------------------------
docdict = dict()

# -------------------------------- general -----------------------------------
docdict['folder_data'] = """
folder : path-like
    Path to the directory containing raw data with recordings, logs and
    models."""
docdict['participant'] = """
participants : int
    ID of the participant to include."""
docdict['participants'] = """
participants : list | tuple
    List of participant IDx to include."""
docdict['session'] = """
session : int
    ID of the session to include."""
docdict['raw'] = """
raw : Raw"""

# --------------------------------- evamed -----------------------------------
docdict['df_raw_evamed'] = """
df : DataFrame
    DataFrame loaded by neurotin.io.read_csv_evamed()."""
docdict['df_clinical'] = """
df: DataFrame
    DataFrame containing the columns 'participant, 'visit', 'date', and
    'result'."""

# ------------------------------ preprocessing -------------------------------
docdict['bads'] = """
bads : list
    List of bad channels."""
docdict['bandpass'] = """
bandpass : tuple
    A 2-length tuple (highpass, lowpass), e.g. (1., 40.).
    The lowpass or highpass filter can be disabled by using None."""
docdict['notch'] = """
notch : bool
    If True, a notch filter at (50, 100, 150) Hz  is applied."""

# -------------------------------- externals ---------------------------------
docdict['plt.figsize'] = """
figsize : tuple
    2-sequence tuple defining the matplotlib figure size."""
docdict['n_jobs'] = """
n_jobs : int
    Number of parallel jobs used. Must not exceed the core count. Can be -1
    to use all cores."""

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