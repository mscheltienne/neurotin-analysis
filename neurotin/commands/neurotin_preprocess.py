import argparse
import multiprocessing as mp
import os

import mne

from neurotin import set_log_level
from neurotin.commands import helpdict
from neurotin.io.cli import write_results
from neurotin.preprocessing import pipeline
from neurotin.utils.list_files import raw_fif_selection
from neurotin.utils._checks import _check_path, _check_n_jobs


def run():
    """Entrypoint for neurotin.preprocessing.pipeline"""
    parser = argparse.ArgumentParser(
        prog='NeuroTin',
        description='NeuroTin auto-preprocessing pipeline.')

    parser.add_argument(
        'dir_in', type=str,
        help='folder where FIF files to process are stored.')

    parser.add_argument(
        'dir_out', type=str,
        help='folder where processed FIF files are saved.')

    parser.add_argument(
        '--n_jobs', type=int, metavar='int', help=helpdict['n_jobs'],
        default=1)

    parser.add_argument(
        '--participant', type=int, metavar='int',
        help='restrict processing to files with this participant ID.',
        default=None)

    parser.add_argument(
        '--session', type=int, metavar='int',
        help='restrict processing to files with this session ID.',
        default=None)

    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='restrict processing to this file.',
        default=None)

    parser.add_argument(
        '--ignore_existing', action='store_true',
        help='ignore files already processed and saved in dir_out.')

    parser.add_argument(
        '--loglevel', type=str, metavar='str', help=helpdict['loglevel'],
        default='info')

    parser.add_argument(
        '--loglevel_mne', type=str, metavar='str',
        help=helpdict['loglevel_mne'], default='error')

    # parse and set log levels
    args = parser.parse_args()
    set_log_level(args.loglevel.upper().strip())
    mne.set_log_level(args.loglevel_mne.upper().strip())

    # check arguments
    dir_in = _check_path(args.dir_in, 'dir_in', must_exist=True)
    dir_out = _check_path(args.dir_out, 'dir_out', must_exist=False)
    n_jobs = _check_n_jobs(args.n_jobs)

    os.makedirs(dir_out, exist_ok=True)

    # list files to process
    fifs = raw_fif_selection(
        dir_in,
        dir_out,
        participant=args.participant,
        session=args.session,
        fname=args.fname,
        ignore_existing=args.ignore_existing
        )

    input_pool = [(fname, dir_in, dir_out) for fname in fifs]
    assert 0 < len(input_pool)  # sanity-check

    # process and save results
    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    write_results(results, dir_out / 'preprocess.pcl')
