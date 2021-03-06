import glob
import os
import sys
from importlib import import_module
from pathlib import Path

import neurotin


def run():
    """Entrypoint for neurotin <command> usage."""
    root = Path(__file__).parent
    valid_commands = sorted(glob.glob(str(root / "neurotin_*.py")))
    valid_commands = [
        file.split(os.path.sep)[-1][:-3] for file in valid_commands
    ]

    def print_help():
        print("Usage: NeuroTin command options\n")
        print("Accepted commands:\n")
        for command in valid_commands:
            print("\t- %s" % command)

    if len(sys.argv) == 1 or "help" in sys.argv[1] or "-h" in sys.argv[1]:
        print_help()
    elif sys.argv[1] == "--version":
        print("NeuroTin-analysis %s" % neurotin.__version__)
    elif sys.argv[1] not in valid_commands:
        print('Invalid command: "%s"\n' % sys.argv[1])
        print_help()
    else:
        cmd = sys.argv[1]
        cmd = import_module(".neurotin_%s" % (cmd,), "neurotin.commands")
        sys.argv = sys.argv[1:]
        cmd.run()
