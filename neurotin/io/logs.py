from datetime import datetime
from pathlib import Path
from typing import Union

from ..utils._checks import _check_path


def read_logs(session_dir: Union[str, Path]):
    """Read logs for a given participant/session.

    Parameters
    ----------
    session_dir : path-like
        Path to the session folder in the raw dataset.
    """
    session_dir = _check_path(
        session_dir, item_name="session_dir", must_exist=True
    )
    logs_file = _check_path(
        session_dir / "logs.txt", item_name="logs_file", must_exist=True
    )
    with open(logs_file, "r") as f:
        lines = f.readlines()
    lines = [line.split(" - ") for line in lines if len(line.split(" - ")) > 1]
    logs = [
        [datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M")]
        + [line[k].strip() for k in range(1, len(line))]
        for line in lines
    ]
    return sorted(logs, key=lambda x: x[0], reverse=False)
