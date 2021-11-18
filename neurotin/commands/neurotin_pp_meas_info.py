import argparse

from neurotin.commands import helpdict
from neurotin.preprocessing.meas_info import _cli


def run():
    """Entrypoint for neurotin.preprocessing.meas_info"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Fill measurement information for NeuroTin raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str, help=helpdict['input_dir_fif'])
    parser.add_argument(
        'output_dir_fif', type=str, help=helpdict['output_dir_fif_with_None'])
    parser.add_argument(
        'raw_dir_fif', type=str,
        help='folder containing raw FIF files with session logs.')
    parser.add_argument(
        'subject_info', type=str,
        help='path to the subject info file.')
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

    output_dir_fif = None if args.output_dir_fif.lower().strip() == 'none' \
        else args.output_dir_fif

    _cli(args.input_dir_fif, output_dir_fif, args.raw_dir_fif,
         args.subject_info, args.n_jobs, args.participant, args.session,
         args.fname, args.ignore_existing)
