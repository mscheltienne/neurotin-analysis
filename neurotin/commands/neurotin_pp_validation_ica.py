import argparse

from neurotin.preprocessing.validation.ica import _cli


def run():
    """Entrypoint for neurotin_logs_mml."""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Checks the scores and components removed by ICA with '
                    'different methods and thresholds.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'result_file', type=str,
        help='path to the file where the results are pickled.')
    parser.add_argument(
        '--n_jobs', type=int, metavar='int',
        help='number of parallel jobs.', default=1)
    parser.add_argument(
        '--subject', type=int, metavar='int',
        help='restrict to files with this subject ID.', default=None)
    parser.add_argument(
        '--session', type=int, metavar='int',
        help='restrict with files with this session ID.', default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='restrict to this file.', default=None)

    args = parser.parse_args()

    _cli(args.input_dir_fif, args.result_file, args.n_jobs, args.subject,
         args.session, args.fname)
