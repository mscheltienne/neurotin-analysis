"""Config module."""

import os
from pathlib import Path
from configparser import ConfigParser


def load_paths():
    """Load paths stored in paths.ini depending on the system used."""
    config = ConfigParser(inline_comment_prefixes=('#', ';'))
    config.optionxform = str
    config.read(Path(__file__).parent/'paths.ini')

    # find which PC is in use
    name = os.getenv('COMPUTERNAME')

    return {key.replace('_', '-').lower(): path
            for key, path in config.items(name)}
