"""Script to apply projectors on raw .fif files."""

import os
import pickle
import traceback
import multiprocessing as mp

import mne

from .list_files import raw_fif_selection
from .. import logger
from ..utils.checks import _check_path, _check_n_jobs

mne.set_log_level('ERROR')


def pipeline(fname, input_dir_fif, output_dir_fif):
    """
    Pipeline function called on each raw file.

    Apply projectors.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to convert.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname).
    output_dir_fif : str | Path | None
        Path used to save raw in MNE format with the same structure as in
        fname. If None, the input file is overwritten in-place.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to convert.
    """
    logger.info('Processing: %s' % fname)
    try:
        fname = _check_path(fname, item_name='fname', must_exist=True)
        input_dir_fif = _check_path(input_dir_fif,
                                    item_name='input_dir_fif',
                                    must_exist=True)
        if output_dir_fif is not None:
            output_dir_fif = _check_path(output_dir_fif,
                                         item_name='output_dir_fif',
                                         must_exist=True)

        relative_fname = fname.relative_to(input_dir_fif)
        # create output file name
        if output_dir_fif is not None:
            output_fname = output_dir_fif / relative_fname
            os.makedirs(output_fname.parent, exist_ok=True)
        else:
            output_fname = fname

        raw = mne.io.read_raw_fif(fname, preload=True)
        raw.apply_proj()
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname))

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname))


def main(input_dir_fif, output_dir_fif, n_jobs=1, participant=None,
         session=None, fname=None, ignore_existing=True):
    """
    CLI processing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files.
    output_dir_fif : str | Path | None
        Path to the folder containing the FIF files processed. If None, the
        input file is overwritten in-place.
    n_jobs : int
        Number of parallel jobs used. Must not exceed the core count. Can be -1
        to use all cores.
    participant : int | None
        Restricts file selection to this participant.
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
    if output_dir_fif is not None:
        output_dir_fif = _check_path(output_dir_fif,
                                     item_name='output_dir_fif')
        os.makedirs(output_dir_fif, exist_ok=True)
    else:
        output_dir_fif = input_dir_fif
        ignore_existing = False  # overwrite existing files
    n_jobs = _check_n_jobs(n_jobs)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude=[],
                                participant=participant, session=session,
                                fname=fname, ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_fif/'fails.pcl', mode='wb') as f:
        pickle.dump([file for success, file in results if not success], f, -1)
