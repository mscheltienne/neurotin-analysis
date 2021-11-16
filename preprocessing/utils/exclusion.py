from pathlib import Path

from checks import _check_path, _check_type


def read_exclusion(exclusion_file):
    """
    Read the list of input fif files to exclude from preprocessing.
    If the file storing the exlusion list does not exist, it is created.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the path to input files to exclude.

    Returns
    -------
    exclude : list
        List of files to exclude.
    """
    exclusion_file = _check_path(exclusion_file, 'exclusion_file')
    if exclusion_file.exists():
        with open(exclusion_file, 'r') as file:
            exclude = file.readlines()
        exclude = [line.rstrip() for line in exclude if len(line) > 0]
    else:
        with open(exclusion_file, 'w'):
            pass
        exclude = list()
    return [Path(file) for file in exclude]


def write_exclusion(exclusion_file, exclude):
    """
    Add a fif file or a set of fif files to the exclusion file.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the path to input files to exclude.
    exclude : str | Path | list | tuple
        Path or list of Paths to input files to exclude.
    """
    exclusion_file = _check_path(exclusion_file, 'exclusion_file')
    _check_type(exclude, ('path-like', list, tuple), 'exclude')
    mode = 'w' if not exclusion_file.exists() else 'a'
    if isinstance(exclude, (str, Path)):
        exclude = [str(exclude)] if Path(exclude).exists() else []
    elif isinstance(exclude, (list, tuple)):
        exclude = [str(fif) for fif in exclude if Path(fif).exists()]
    with open(exclusion_file, mode) as file:
        for fif in exclude:
            file.write(str(fif) + '\n')
