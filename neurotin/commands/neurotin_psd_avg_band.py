import argparse
import pickle

import mne

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.psd import psd_avg_band


def run():
    """Entrypoint for neurotin.time_frequency.psd.compute_psd_average_bins"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='Compute PSD from online recordings.')

    parser.add_argument(
        'dir_in', type=str,
        help='folder where preprocessed FIF files are saved.')

    parser.add_argument(
        'df_fname', type=str, help='path where the dataframe is pickled.')

    parser.add_argument(
        '-p', '--participants', help='participant ID(s) to include.',
        nargs='+', required=True)

    parser.add_argument(
        '-d', '--duration', type=int, metavar='int',
        help='duration of epochs for welch method.',
        required=True)

    parser.add_argument(
        '-o', '--overlap', type=int, metavar='int',
        help='overlap duration between epochs for welch method.',
        required=True)

    parser.add_argument(
        '--reject', action='store_true',
        help='flag to reject epochs for welch method with autoreject')

    parser.add_argument(
        '--fmin', type=int, metavar='int',
        help='minimum frequency of interest.', required=True)

    parser.add_argument(
        '--fmax', type=int, metavar='int',
        help='maximum frequency of interest.', required=True)

    parser.add_argument(
        '-a', '--average', type=str, metavar='str',
        help='average method between frequency bins.', default='integrate')

    parser.add_argument(
        '--n_jobs', type=int, metavar='int', help=helpdict['n_jobs'],
        default=1)

    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    parser.add_argument(
        '--loglevel_mne', type=str, metavar='str',
        help=helpdict['loglevel_mne'], default='error')

    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())
    mne.set_log_level(args.loglevel_mne.upper().strip())

    # assert result file is writable
    try:
        with open(args.df_fname, 'wb') as f:
            pickle.dump('data will be written here..', f, -1)
    except Exception:
        raise IOError("Could not write to file: '%s'." % args.result_file)

    participants = [int(participant) for participant in args.participants]
    reject = 'auto' if args.reject else None

    df = psd_avg_band(
        args.dir_in,
        participants,
        args.duration,
        args.overlap,
        reject,
        args.fmin,
        args.fmax,
        args.average,
        args.n_jobs
        )
    df.to_pickle(args.df_fname, compression=None)
