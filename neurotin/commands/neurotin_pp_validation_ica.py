import argparse

from neurotin.commands import helpdict
from neurotin.preprocessing.validation.ica import _cli


def run():
    """Entrypoint for neurotin.preprocessing.validation.ica"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Checks the scores and components removed by ICA with '
                    'different methods and thresholds.')
    parser.add_argument(
        'input_dir_fif', type=str, help=helpdict['input_dir_fif'])
    parser.add_argument(
        'result_file', type=str,
        help='path to the file where the test results are pickled.')
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

    args = parser.parse_args()

    _cli(args.input_dir_fif, args.result_file, args.n_jobs, args.participant,
         args.session, args.fname)
