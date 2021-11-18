import argparse

from neurotin.commands import helpdict
from neurotin.preprocessing.prepare_raw import _cli


def run():
    """Entrypoint for neurotin.preprocessing.prepare_raw"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Prepare NeuroTin raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str, help=helpdict['input_dir_fif'])
    parser.add_argument(
        'output_dir_fif', type=str, help=helpdict['output_dir_fif'])
    parser.add_argument(
        '--n_jobs', type=int, metavar='int', help=helpdict['n_jobs'],
        default=1)
    parser.add_argument(
        '--participant', type=int, metavar='int', help=helpdict['participant'],
        default=None)
    parser.add_argument(
        '--session', type=int, metavar='int', help=helpdict['session'],
        default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path', help=helpdict['fname'],
        default=None)
    parser.add_argument(
        '--ignore_existing', action='store_true',
        help=helpdict['ignore_existing'])

    args = parser.parse_args()

    _cli(args.input_dir_fif, args.output_dir_fif, args.n_jobs,
         args.participant, args.session, args.fname, args.ignore_existing)
