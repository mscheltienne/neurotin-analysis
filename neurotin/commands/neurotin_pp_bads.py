import argparse

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.preprocessing.bads import _cli


def run():
    """Entrypoint for neurotin.preprocessing.bads"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Prepare NeuroTin raw FIF files.')
    parser.add_argument(
        'dir_in', type=str, help=helpdict['input_dir_fif'])
    parser.add_argument(
        'dir_out', type=str, help=helpdict['output_dir_fif'])
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
    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())

    _cli(
        args.dir_in,
        args.dir_out,
        args.n_jobs,
        args.participant,
        args.session,
        args.fname,
        args.ignore_existing)
