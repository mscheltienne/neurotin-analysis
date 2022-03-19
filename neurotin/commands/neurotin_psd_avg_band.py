import pickle
import argparse

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.time_frequency.psd import compute_psd_average_bins


def run():
    """Entrypoint for neurotin.time_frequency.psd.compute_psd_average_bins"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Compute PSD from online recordings.')
    parser.add_argument(
        'input_dir_fif', type=str, help=helpdict['input_dir_fif'])
    parser.add_argument(
        'result_file', type=str, help='path where the dataframe is pickled.')
    parser.add_argument(
        '-p', '--participants',
        help=helpdict['participants'], nargs='+', required=True)
    parser.add_argument(
        '-d', '--duration', type=int, metavar='int',
        help='epochs duration on which welch is applied.',
        required=True)
    parser.add_argument(
        '-o', '--overlap', type=int, metavar='int',
        help='epochs overlap on which welch is applied.',
        required=True)
    parser.add_argument(
        '--reject', action='store_true',
        help='flag to reject epochs with autoreject')
    parser.add_argument(
        '--fmin', type=int, metavar='int',
        help='minimum frequency of interest.', required=True)
    parser.add_argument(
        '--fmax', type=int, metavar='int',
        help='maximum frequency of interest.', required=True)
    parser.add_argument(
        '-a', '--average', type=str, metavar='str',
        help='average method between frequency bins.', default='mean')
    parser.add_argument(
        '--n_jobs', type=int, metavar='int', help=helpdict['n_jobs'],
        default=1)
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

    participants = [int(participant) for participant in args.participants]
    reject = 'auto' if args.reject else None

    df = compute_psd_average_bins(args.input_dir_fif, participants,
                                  args.duration, args.overlap, reject,
                                  args.fmin, args.fmax, args.average,
                                  args.n_jobs)
    df.to_pickle(args.result_file, compression=None)
