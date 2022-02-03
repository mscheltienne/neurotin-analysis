import re
import pickle
import argparse
from pathlib import Path

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.model import compute_average


def run():
    """Entrypoint for neurotin.model.compute_average"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Compute average model from online recordings.')
    parser.add_argument(
        'input_dir', type=str,
        help='folder where raw data and models is stored')
    parser.add_argument(
        'result_file', type=str, help='path where the dataframe is pickled.')
    parser.add_argument(
        '-p', '--participants',
        help=helpdict['participants'], nargs='+', required=False)
    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())

    # assert result file is writable
    try:
        with open(args.result_file, 'wb') as f:
            pickle.dump('data will be written here..', f, -1)
    except Exception:
        raise IOError("Could not write to file: '%s'." % args.result_file)

    if args.participants is not None:
        participants = [int(participant) for participant in args.participants]
    else:
        pattern = re.compile(r'(\d{3})')
        participants = [int(p.name) for p in Path(args.input_dir).iterdir()
                        if pattern.search(str(p))]

    if len(participants) == 0:
        raise ValueError('Could not find any participants to merge.')

    df = compute_average(args.input_dir, participants)
    df.to_pickle(args.result_file, compression=None)
