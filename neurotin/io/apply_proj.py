"""Script to apply projectors on raw .fif files."""

import os
import traceback
import multiprocessing as mp

import mne

from .cli_results import write_results
from .list_files import raw_fif_selection
from .. import logger
from ..utils.docs import fill_doc
from ..utils.checks import _check_path, _check_n_jobs

mne.set_log_level('ERROR')


@fill_doc
def _pipeline(fname, input_dir_fif, output_dir_fif):
    """%(pipeline_header)s

    Apply projectors.

    Parameters
    ----------
    %(fname)s
    %(input_dir_fif)s
    %(output_dir_fif_with_None)s

    Returns
    -------
    %(success)s
    %(fname)s
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


@fill_doc
def _cli(input_dir_fif, output_dir_fif, n_jobs=1, participant=None,
         session=None, fname=None, ignore_existing=True):
    """%(cli_header)s

    Parameters
    ----------
    %(input_dir_fif)s
    %(output_dir_fif_with_None)s
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    %(ignore_existing)s
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

    # list files to process
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude=[],
                                participant=participant, session=session,
                                fname=fname, ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    write_results(results, output_dir_fif/'apply_proj.pcl')
