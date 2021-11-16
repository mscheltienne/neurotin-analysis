"""Script to convert RAW .fif files to RAW .set files for EEGLAB."""

import os
import pickle
import argparse
import traceback
import multiprocessing as mp

import mne

from utils.list_files import raw_fif_selection
from utils.checks import _check_path, _check_n_jobs

mne.set_log_level('ERROR')


def pipeline(fname, input_dir_fif, output_dir_set):
    """
    Pipeline function called on each raw file.

    Convert .fif to .set files.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to convert.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname).
    output_dir_set : str | Path
        Path used to save raw in EEGLAB format with the same structure as in
        fname.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to convert.
    """
    try:
        fname = _check_path(fname, 'fname', must_exist=True)
        input_dir_fif = _check_path(input_dir_fif,
                                    item_name='input_dir_fif',
                                    must_exist=True)
        output_dir_set = _check_path(output_dir_set,
                                     item_name='output_dir_set',
                                     must_exist=True)

        relative_fname = fname.relative_to(input_dir_fif)
        output_fname_set = output_dir_set / relative_fname.with_suffix('.set')
        os.makedirs(output_fname_set.parent, exist_ok=True)

        raw = mne.io.read_raw_fif(fname, preload=True)
        raw.apply_proj()
        raw.export(output_fname_set, fmt='eeglab')

        return (True, str(fname))

    except Exception:
        print ('----------------------------------------------')
        print ('FAILED: %s -> Skip.' % fname)
        print(traceback.format_exc())
        print ('----------------------------------------------')
        return (False, str(fname))


def main(input_dir_fif, output_dir_set, n_jobs=1, subject=None, session=None,
         fname=None, ignore_existing=True):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files.
    output_dir_set : str | Path
        Path to the folder containing the EEGLAB files.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).
    ignore_existing : bool
        If True, files already converted are not included.
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, item_name='input_dir_fif',
                                must_exist=True)
    output_dir_set = _check_path(output_dir_set, item_name='output_dir_set')
    n_jobs = _check_n_jobs(n_jobs)

    # create output folder if needed
    os.makedirs(output_dir_set, exist_ok=True)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_set, exclude=[],
                                subject=subject, session=session, fname=fname)

    # create input pool for pipeline based on provided subject info
    input_pool = [(fname, input_dir_fif, output_dir_set)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_set/'fails.pcl', mode='wb') as f:
        pickle.dump([file for success, file in results if not success], f, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin converting to set patch.',
        description='Convert raw to set.')
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

    main(args.input_dir_fif, args.output_dir_set, args.n_jobs, args.subject,
         args.session, args.fname, args.ignore_existing)