"""Utility functions for checking types and values. Inspired from MNE."""

import os
import operator
from pathlib import Path
import multiprocessing

import numpy as np

from .. import logger


def _ensure_int(item, *, item_name=None):
    """
    Ensure a variable is an integer.

    Parameters
    ----------
    item : object
        Item to check.
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not int.
    """
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(item, bool):
            raise TypeError
        item = int(operator.index(item))
    except TypeError:
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError("%s must be an int, got %s instead."
                        % (item_name, type(item)))


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable:
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_types = {
    'numeric': (np.floating, float, _IntLike()),
    'path-like': (str, Path, os.PathLike),
    'int': (_IntLike(), ),
    'callable': (_Callable(), ),
}


def _check_type(item, types, *, item_name=None):
    """
    Check that item is an instance of types.

    Parameters
    ----------
    item : object
        Item to check.
    types : tuple of types | tuple of str
        Types to be checked against.
        If str, must be one of:
            ('int', 'str', 'numeric', 'path-like', 'callable')
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """
    check_types = sum(((type(None), ) if type_ is None else (type_, )
                       if not isinstance(type_, str) else _types[type_]
                       for type_ in types), ())

    if not isinstance(item, check_types):
        type_name = ["None" if cls_ is None else cls_.__name__
                     if not isinstance(cls_, str) else cls_
                     for cls_ in types]
        if len(type_name) == 1:
            type_name = type_name[0]
        elif len(type_name) == 2:
            type_name = ' or '.join(type_name)
        else:
            type_name[-1] = "or " + type_name[-1]
            type_name = ", ".join(type_name)
        item_name = "Item" if item_name is None else "'%s'" % item_name
        raise TypeError(f"{item_name} must be an instance of {type_name}, "
                        f"got {type(item)} instead.")


def _check_value(item, allowed_values, *, item_name=None, extra=None):
    """
    Check the value of a parameter against a list of valid options.

    Parameters
    ----------
    item : object
        Item to check.
    allowed_values : tuple of objects
        Allowed values to be checked against.
    item_name : str | None
        Name of the item to show inside the error message.
    extra : str | None
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """
    if item not in allowed_values:
        item_name = "" if item_name is None else " '%s'" % item_name
        extra = "" if extra is None else " " + extra
        msg = ("Invalid value for the{item_name} parameter{extra}. "
               '{options}, but got {item!r} instead.')
        allowed_values = tuple(allowed_values)  # e.g., if a dict was given
        if len(allowed_values) == 1:
            options = "The only allowed value is %s" % repr(allowed_values[0])
        elif len(allowed_values) == 2:
            options = "Allowed values are %s and %s" % \
                (repr(allowed_values[0]), repr(allowed_values[1]))
        else:
            options = "Allowed values are "
            options += ", ".join([f"{repr(v)}" for v in allowed_values[:-1]])
            options += f", and {repr(allowed_values[-1])}"
        raise ValueError(msg.format(item_name=item_name, extra=extra,
                                    options=options, item=item))


def _check_path(item, *, item_name=None, must_exist=False):
    """Check if path is a valid and return it as pathlib.Path.

    Parameters
    ----------
    item : object
        Item to check.
    item_name : str | None
        Name of the item to show inside the error message.
    must_exist : bool
        If True, the path must lead to an existing directory or file.

    Returns
    -------
    item : Path
        Provided path as a pathlib.Path object.

    Raises
    ------
    TypeError
        When item is not a path-like object.
    AssertionError
        When must_exist is set to True and the path does not lead to an
        existing directory or file.
    """
    _check_type(item, ('path-like', ), item_name=item_name)
    item = Path(item)
    logger.debug("Checking path '%s'.", item)
    if must_exist:
        assert item.exists(), 'The path does not exists.'
    return item


def _check_n_jobs(n_jobs):
    """Checks that the number of jobs is valid.

    Parameters
    ----------
    n_jobs : int
        Number of jobs to run. Must be smaller than the number of cores on the
        computer. Can be -1 to use all available cores.

    Returns
    -------
    n_jobs : int
        Number of jobs to run.

    Notes
    -----
    Number of cores is retrieved with multiprocessing.cpu_count(). Behavior
    may differ based on the OS.
    """
    _check_type(n_jobs, ('int', ), item_name='n_jobs')
    n_cores = multiprocessing.cpu_count()
    logger.debug("Checking n_jobs '%i'.", n_jobs)
    logger.debug("Number of cores found: %s", n_cores)
    if n_jobs == -1:
        n_jobs = n_cores
    _check_value(n_jobs, tuple(range(n_cores+1)), item_name='n_jobs')
    return n_jobs


def _check_participant(participant):
    """Checks that the participant ID is valid.

    Parameters
    ----------
    participant : int
        Participant ID.

    Raises
    ------
    TypeError
        When the participant ID is not an int.
    AssertionError
        When the participant ID is not a strictly positive integer.
    """
    _check_type(participant, ('int', ), item_name='participant')
    assert 0 < participant
    return participant


def _check_participants(participants):
    """Checks that the participant IDs are valid and return them as a list.

    Parameters
    ----------
    participants : int | list | tuple
        Participant ID(s).

    Returns
    -------
    participants : list
        Participant ID(s).

    Raises
    ------
    TypeError
        When the participant ID is not an int.
    AssertionError
        When the participant ID is not a strictly positive integer.
    """
    _check_type(participants, ('int', list, tuple), item_name='participants')

    if isinstance(participants, (int, )):
        participants = [participants]
    elif isinstance(participants, (tuple, )):
        participants = list(participants)

    for participant in participants:
        _check_participant(participant)

    return participants


def _check_session(session):
    """Checks that the session ID is valid.

    Parameters
    ----------
    session : int
        Session ID.

    Raises
    ------
    TypeError
        When the session ID is not an int.
    AssertionError
        When the session ID is not between 1 and 15 included.
    """
    _check_type(session, ('int', ), item_name='session')
    assert 1 <= session <= 15
    return session
