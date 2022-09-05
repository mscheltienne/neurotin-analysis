import argparse
import pickle
import re
from pathlib import Path

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.model import compute_average
from neurotin.utils._checks import _check_path


def run():
    """Entrypoint for 'neurotin.model.compute_average'."""
    parser = argparse.ArgumentParser(
        prog="NeuroTin",
        description="Compute average model from online recordings.",
    )
    parser.add_argument(
        "dir_in", type=str, help="folder where raw data and models are stored."
    )
    parser.add_argument(
        "df_fname", type=str, help="path where the dataframe is pickled."
    )
    parser.add_argument(
        "-p",
        "--participants",
        help="participant ID(s) to include.",
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        metavar="str",
        help=helpdict["loglevel"],
        default="info",
    )

    # parse and set log levels
    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())

    dir_in = _check_path(args.dir_in, "dir_in", must_exist=True)

    # assert result file is writable
    try:
        with open(args.df_fname, "wb") as f:
            pickle.dump("data will be written here..", f, -1)
    except Exception:
        raise IOError(f"Could not write to file: '{args.df_fname}'.")

    if args.participants is not None:
        participants = [int(participant) for participant in args.participants]
    else:
        pattern = re.compile(r"(\d{3})")
        participants = [
            int(p.name)
            for p in Path(args.dir_in).iterdir()
            if pattern.match(p.name)
        ]

    if len(participants) == 0:
        raise ValueError("Could not find any participants to merge.")

    df = compute_average(dir_in, participants)
    df.to_pickle(args.df_fname, compression=None)
