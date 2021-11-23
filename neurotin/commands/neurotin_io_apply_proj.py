import argparse

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.io.apply_proj import _cli


def run():
    """Entrypoint for neurotin.io.apply_proj"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Apply projector to raw files.')
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
    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())

    output_dir_fif = None if args.output_dir_fif.lower().strip() == 'none' \
        else args.output_dir_fif
    _cli(args.input_dir_fif, output_dir_fif, args.n_jobs, args.participant,
         args.session, args.fname, args.ignore_existing)
