from pathlib import Path

from checks import _check_type, _check_path


def list_raw_fif(directory, exclude=[]):
    """
    List all raw fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory.
    exclude : list | tuple
        List of files to exclude.

    Returns
    -------
    fifs : list
        Found raw fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_raw_fif(elt))
        elif elt.name.endswith("-raw.fif") and elt not in exclude:
            fifs.append(elt)
    return fifs


def list_ica_fif(directory):
    """
    List all ica fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory.

    Returns
    -------
    fifs : list
        Found ica fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_ica_fif(elt))
        elif elt.name.endswith("-ica.fif"):
            fifs.append(elt)
    return fifs


def raw_fif_selection(input_dir_fif, output_dir_fif, exclude, *, subject,
                      session, fname):
    """List raw fif file to preprocess.

    The list of files is filtered by exclusion/subject/session. Files already
    present in output_dir_fif are ignored.
    If fname is provided, other conditions don't apply and [fname] is returned.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files to preprocess.
    output_dir_fif : str | Path
        Path to the folder containing the FIF files preprocessed.
    exclude : list
        List of pathlib.Path instance to exclude from the selection.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).

    Returns
    -------
    fifs_in : list
        List of file(s) selected.
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, 'input_dir_fif',
                                must_exist=True)
    output_dir_fif = _check_path(output_dir_fif, 'output_dir_fif')
    exclude = _check_type(exclude, (list, ), 'exclude')
    subject = _check_subject(subject)
    session = _check_session(session)
    fname = _check_fname(fname, input_dir_fif)

    # list files
    fifs_in = [f for f in list_raw_fif(input_dir_fif, exclude=exclude)
               if not (output_dir_fif/f.relative_to(input_dir_fif)).exists()]
    subjects = [int(file.parent.parent.parent.name) for file in fifs_in]
    sessions = [int(file.parent.parent.name.split()[1]) for file in fifs_in]

    # filter
    if subject is not None:
        sessions = [session_id for k, session_id in enumerate(sessions)
                    if subjects[k] == subject]
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if subjects[k] == subject]
    if session is not None:
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if sessions[k] == session]
    if fname is not None:
        assert fname in fifs_in
        fifs_in = [fname]

    return fifs_in


def _check_subject(subject):
    """Checks that the subject ID is valid."""
    _check_type(subject, (None, 'int'), 'subject')
    if subject is not None:
        assert 0 < subject, 'subject should be a positive integer'
    return subject


def _check_session(session):
    """Checks that the session ID is valid."""
    _check_type(session, (None, 'int'), 'session')
    if session is not None:
        assert 1 <= session <= 15, 'session should be included in (1, 15)'
    return session


def _check_fname(fname, folder):
    """Checks that the fname is valid and present in folder."""
    fname = _check_path(fname)
    folder = _check_path(folder, must_exist=True)
    if fname is not None:
        fname.relative_to(folder)  # raise if fname is not in folder
    return fname
