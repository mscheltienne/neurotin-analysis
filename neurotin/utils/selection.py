import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ..io.logs import read_logs
from ._checks import _check_participants, _check_path, _check_type
from ._docs import fill_doc
from ._logs import logger


@fill_doc
def list_runs(
    folder: Union[str, Path],
    participants: Union[int, List[int], Tuple[int, ...]],
    valid_only: bool = True,
    regular_only: bool = False,
    transfer_only: bool = False,
) -> Dict[int, Dict[int, str]]:
    """List online neurofeedback runs.

    Parameters
    ----------
    %(folder_raw_data)s
    %(participants)s
    %(valid_only)s
    %(regular_only)s
    %(transfer_only)s

    Returns
    -------
    runs : dict
        Dictionary containing the selected runs.
        Key: int - participant ID.
        Value: dictionnary
            Key: int - session ID.
            Value: str - list of online runs file path.
    """
    folder = _check_path(folder, "folder", must_exist=True)
    participants = _check_participants(participants)
    _check_type(valid_only, (bool,), "valid_only")
    _check_type(regular_only, (bool,), "regular_only")
    _check_type(transfer_only, (bool,), "transfer_only")
    assert not (regular_only and transfer_only)

    # participant folder pattern
    pattern = re.compile(r"(\d{3})")
    participants_folder = [
        int(p.name) for p in folder.iterdir() if pattern.match(p.name)
    ]

    runs = dict()
    # list valid runs
    for participant in participants:
        if participant not in participants_folder:
            logger.warning(
                "Participant %i is missing from provided folder.", participant
            )
            continue
        if participant not in runs:
            runs[participant] = dict()

        for session in range(1, 16):
            if session not in runs[participant]:
                runs[participant][session] = list()

            session_dir = (
                folder / str(participant).zfill(3) / f"Session {session}"
            )
            # read and filter the logs
            logs = read_logs(session_dir)
            onRun = [log for log in logs if log[1] == "OnRun"]
            if valid_only:
                onRun = [log for log in onRun if len(log) == 3]
            if regular_only:
                onRun = [log for log in onRun if "neurofeedback" in log[2]]
            if transfer_only:
                onRun = [log for log in onRun if "transfer" in log[2]]
            del logs

            # list online run files
            files = [
                file
                for file in (session_dir / "Online").iterdir()
                if file.is_file() and file.suffix == ".fif"
            ]
            for file in files:
                if not file.name[0].isdigit():
                    logger.error("Unexpected file %s", file)

            # retrieve files corresponding to the filtered logs
            idx = [int(log[2][-1]) for log in onRun]
            for file in files:
                if int(file.name[0]) in idx:
                    runs[participant][session].append(str(file))

            # sort runs
            runs[participant][session] = sorted(runs[participant][session])

    return runs


@fill_doc
def list_runs_pp(
    folder: Union[str, Path],
    folder_pp: Union[str, Path],
    participants: Union[int, List[int], Tuple[int, ...]],
    valid_only: bool = True,
    regular_only: bool = False,
    transfer_only: bool = False,
) -> Dict[int, Dict[int, str]]:
    """List online neurofeedback runs preprocessed.

    Parameters
    ----------
    %(folder_raw_data)s
    %(folder_pp_data)s
    %(participants)s
    %(valid_only)s
    %(regular_only)s
    %(transfer_only)s

    Returns
    -------
    runs : dict
        Dictionary containing the selected runs.
        Key: int - participant ID.
        Value: dictionnary
            Key: int - session ID.
            Value: str - list of online runs file path.
    """
    folder_pp = _check_path(folder_pp, "folder_pp", must_exist=True)
    runs = list_runs(
        folder, participants, valid_only, regular_only, transfer_only
    )

    for participant in runs:
        for session in runs[participant]:
            files_pp = list()
            for file in runs[participant][session]:
                file_pp = folder_pp / Path(file).relative_to(folder)
                if file_pp.exists():
                    files_pp.append(file_pp)
            runs[participant][session] = sorted(files_pp)

    return runs
