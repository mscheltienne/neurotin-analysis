"""Script to convert RAW .fif files to RAW .set files for EEGLAB."""

import os
import pickle
import traceback
import multiprocessing as mp

import mne

from .list_files import raw_fif_selection
from .. import logger
from ..utils.docs import fill_doc
from ..utils.checks import _check_path, _check_n_jobs

mne.set_log_level('ERROR')


@fill_doc
def pipeline(fname, input_dir_fif, output_dir_set):
    """%(pipeline_header)s

    Convert .fif to .set files.

    Parameters
    ----------
    %(fname)s
    %(input_dir_fif)s
    %(output_dir_set)s

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
        output_dir_set = _check_path(output_dir_set,
                                     item_name='output_dir_set',
                                     must_exist=True)

        relative_fname = fname.relative_to(input_dir_fif)
        output_fname_set = output_dir_set / relative_fname.with_suffix('.set')
        os.makedirs(output_fname_set.parent, exist_ok=True)

        raw = mne.io.read_raw_fif(fname, preload=True)
        raw.apply_proj()
        raw.export(str(output_fname_set), fmt='eeglab')

        return (True, str(fname))

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname))


@fill_doc
def main(input_dir_fif, output_dir_set, n_jobs=1, participant=None,
         session=None, fname=None, ignore_existing=True):
    """%(main_header)s

    Parameters
    ----------
    %(input_dir_fif)s
    %(output_dir_set)s
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    %(ignore_existing)s
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
                                participant=participant, session=session,
                                fname=fname, ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, input_dir_fif, output_dir_set)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_set/'fails.pcl', mode='wb') as f:
        pickle.dump([file for success, file in results if not success], f, -1)
