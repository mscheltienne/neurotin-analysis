from ._checks import (_check_type, _check_path, _check_participant,
                       _check_session)


def list_raw_fif(directory, *, exclude=None):
    """
    List all raw fif files in directory and its sub-directories.

    Parameters
    ----------
    directory : path-like
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
    List all ICA fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : path-like
        Path to the directory where ICA files are searched.
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


def raw_fif_selection(
        dir_in,
        dir_out,
        *,
        participant=None,
        session=None,
        fname=None,
        ignore_existing: bool = True
        ):
    """List raw fif file to process. The list of files can be filtered by
    participant / session / fname.

    Parameters
    ----------
    dir_in : path-like
        Path to the folder containing the FIF files to process
    dir_out : path-like
        Path to the folder containing the FIF files processed. The FIF files
        are saved under the same relative folder structure as in 'dir_in'.
    participant : int | None
        If not None, restricts processing to this participant.
    session : int | None
        If not None, restricts processing to this session.
    fname : path-like | None
        If not None, restricts processing to this file.
    ignore_existing : bool
        If True, files already processed and saved in 'dir_out' are ignored.

    Returns
    -------
    fifs : list
        List of file(s) selected.
    """
    # check arguments
    dir_in = _check_path(dir_in, item_name='dir_in', must_exist=True)
    dir_out = _check_path(dir_out, item_name='dir_out')
    participant = participant if participant is None  \
        else _check_participant(participant)
    session = session if session is None else _check_session(session)
    fname = _check_fname(fname, dir_in)

    # list files
    fifs = list_raw_fif(dir_in)
    if ignore_existing:
        fifs = [file for file in fifs
                if not (dir_out / file.relative_to(dir_in)).exists()]
    participants = [int(file.parent.parent.parent.name) for file in fifs]
    sessions = [int(file.parent.parent.name.split()[1]) for file in fifs]

    # filter
    if participant is not None:
        sessions = [session_id for k, session_id in enumerate(sessions)
                    if participants[k] == participant]
        fifs = [file for k, file in enumerate(fifs)
                if participants[k] == participant]
    if session is not None:
        fifs = [file for k, file in enumerate(fifs) if sessions[k] == session]
    if fname is not None:
        assert fname in fifs
        fifs = [fname]

    return fifs


def _check_fname(fname, folder):
    """Checks that the fname is valid and present in folder."""
    if fname is None:
        return fname
    fname = _check_path(fname, must_exist=True)
    folder = _check_path(folder, must_exist=True)
    fname.relative_to(folder)  # raise if fname is not in folder
    return fname
