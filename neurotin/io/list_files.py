from ..utils.checks import (_check_type, _check_path, _check_participant,
                            _check_session)


def list_raw_fif(directory, *, exclude=None):
    """
    List all raw fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory where raw files are searched.
    exclude : list | tuple
        List of files to exclude.

    Returns
    -------
    fifs : list
        Found raw fif files.
    """
    exclude = [] if exclude is None else exclude
    directory = _check_path(directory, item_name='directory', must_exist=True)
    _check_type(exclude, (list, tuple), item_name='exclude')
    for file in exclude:
        _check_path(file, must_exist=False)
    return _list_fif(directory, exclude, endswith='-raw.fif')


def list_ica_fif(directory, *, exclude=None):
    """
    List all ica fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory where ica files are searched.
    exclude : list | tuple
        List of files to exclude.

    Returns
    -------
    fifs : list
        Found ica fif files.
    """
    exclude = [] if exclude is None else exclude
    directory = _check_path(directory, item_name='directory', must_exist=True)
    _check_type(exclude, (list, tuple), item_name='exclude')
    for file in exclude:
        _check_path(file, must_exist=False)
    return _list_fif(directory, exclude, endswith='-ica.fif')


def _list_fif(directory, exclude, endswith):
    """Recursive function listing fif files in directory and its
    subdirectories."""
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(_list_fif(elt, exclude, endswith))
        elif elt.name.endswith(endswith) and elt not in exclude:
            fifs.append(elt)
    return fifs


def raw_fif_selection(input_dir, output_dir, *, participant=None, session=None,
                      fname=None, ignore_existing=True):
    """List raw fif file to process.

    The list of files is filtered by participant/session/fname.

    Parameters
    ----------
    input_dir : str | Path
        Path to the folder containing the FIF files to process.
    output_dir : str | Path
        Path to the folder containing the FIF files processed.
    participant : int | None
        Restricts file selection to this participant.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).
    ignore_existing : bool
        If True, existing output files are not included.

    Returns
    -------
    fifs_in : list
        List of file(s) selected.
    """
    # check arguments
    input_dir = _check_path(input_dir, item_name='input_dir', must_exist=True)
    output_dir = _check_path(output_dir, item_name='output_dir')
    participant = participant if participant is None  \
        else _check_participant(participant)
    session = session if session is None else _check_session(session)
    fname = _check_fname(fname, input_dir)

    # list files
    fifs_in = list_raw_fif(input_dir)
    if ignore_existing:
        fifs_in = [file for file in fifs_in
                   if not (output_dir / file.relative_to(input_dir)).exists()]
    participants = [int(file.parent.parent.parent.name) for file in fifs_in]
    sessions = [int(file.parent.parent.name.split()[1]) for file in fifs_in]

    # filter
    if participant is not None:
        sessions = [session_id for k, session_id in enumerate(sessions)
                    if participants[k] == participant]
        fifs_in = [file for k, file in enumerate(fifs_in)
                   if participants[k] == participant]
    if session is not None:
        fifs_in = [file for k, file in enumerate(fifs_in)
                   if sessions[k] == session]
    if fname is not None:
        assert fname in fifs_in
        fifs_in = [fname]

    return fifs_in


def _check_fname(fname, folder):
    """Checks that the fname is valid and present in folder."""
    if fname is None:
        return fname
    fname = _check_path(fname, must_exist=True)
    folder = _check_path(folder, must_exist=True)
    fname.relative_to(folder)  # raise if fname is not in folder
    return fname
