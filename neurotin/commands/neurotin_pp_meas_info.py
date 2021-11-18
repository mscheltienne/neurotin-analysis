import argparse

from neurotin.preprocessing.meas_info import _cli


def run():
    """Entrypoint for neurotin_logs_mml."""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Fill measurement information for NeuroTin raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_fif', type=str,
        help='folder containing FIF files preprocessed (can be None to '
             'overwrite existing files in input_dir_fif).')
    parser.add_argument(
        'raw_dir_fif', type=str,
        help='folder containing raw FIF files with session logs.')
    parser.add_argument(
        'subject_info', type=str,
        help='path to the subject info file.')
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
    parser.add_argument(
        '--ignore_existing', action='store_true',
        help='ignore files already processed (set to False if output_dir_fif '
             'is set to None).')

    args = parser.parse_args()

    output_dir_fif = None if args.output_dir_fif.lower().strip() == 'none' \
        else args.output_dir_fif

    _cli(args.input_dir_fif, output_dir_fif, args.raw_dir_fif,
         args.subject_info, args.n_jobs, args.subject, args.session,
         args.fname, args.ignore_existing)
