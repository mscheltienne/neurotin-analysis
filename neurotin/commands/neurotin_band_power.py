import argparse
import pickle
from pathlib import Path

import mne

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.time_frequency import compute_bandpower


def run():
    """Entrypoint for 'neurotin.time_frequency.compute_bandpower'."""
    parser = argparse.ArgumentParser(
        prog="NeuroTin",
        description="Compute band-power from online recordings.",
    )
    parser.add_argument(
        "dir_raw",
        type=str,
        help="folder where raw FIF files are saved.",
    )
    parser.add_argument(
        "dir_pp",
        type=str,
        help="folder where preprocessed FIF files are saved.",
    )
    parser.add_argument(
        "df_fname", type=str, help="path where the dataframe is pickled."
    )
    parser.add_argument(
        "--valid_only",
        help="select only valid online run.",
        action="store_true",
    )
    parser.add_argument(
        "--regular_only",
        help="select only regular online run.",
        action="store_true",
    )
    parser.add_argument(
        "--transfer_only",
        help="select only transfer online run.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--participants",
        help="participant ID(s) to include.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        metavar="int",
        help="duration of epochs for welch method.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=int,
        metavar="int",
        help="overlap duration between epochs for welch method.",
        required=True,
    )
    parser.add_argument(
        "--fmin",
        type=int,
        metavar="int",
        help="minimum frequency of interest.",
        required=True,
    )
    parser.add_argument(
        "--fmax",
        type=int,
        metavar="int",
        help="maximum frequency of interest.",
        required=True,
    )
    parser.add_argument(
        "--n_jobs", type=int, metavar="int", help=helpdict["n_jobs"], default=1
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        metavar="str",
        help=helpdict["loglevel"],
        default="info",
    )
    parser.add_argument(
        "--loglevel_mne",
        type=str,
        metavar="str",
        help=helpdict["loglevel_mne"],
        default="error",
    )

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())
    mne.set_log_level(args.loglevel_mne.upper().strip())

    # assert result file is writable
    df_fname = Path(args.df_fname)
    assert df_fname.suffix == ".pcl"
    df_fname_abs = df_fname.with_stem(df_fname.stem + "-abs")
    df_fname_rel = df_fname.with_stem(df_fname.stem + "-rel")
    for fname in (df_fname_abs, df_fname_rel):
        try:
            with open(fname, "wb") as f:
                pickle.dump("data will be written here..", f, -1)
        except Exception:
            raise IOError("Could not write to file: '%s'." % fname)

    participants = [int(participant) for participant in args.participants]

    df_abs, df_rel = compute_bandpower(
        args.dir_raw,
        args.dir_pp,
        args.valid_only,
        args.regular_only,
        args.transfer_only,
        participants,
        args.duration,
        args.overlap,
        args.fmin,
        args.fmax,
        args.n_jobs,
    )
    df_abs.to_pickle(df_fname_abs, compression=None)
    df_rel.to_pickle(df_fname_rel, compression=None)
