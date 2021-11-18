import argparse

from neurotin.io.convert2eeglab import _cli


def run():
    """Entrypoint for neurotin_logs_mml."""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Convert MNE raw files to EEGLAB raw files.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_set', type=str,
        help='folder containing EEGLAB files preprocessed.')
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
        help='ignore files already processed.')

    args = parser.parse_args()

    _cli(args.input_dir_fif, args.output_dir_set, args.n_jobs, args.subject,
         args.session, args.fname, args.ignore_existing)
