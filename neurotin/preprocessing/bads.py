import multiprocessing as mp
import os
import traceback

import mne

from .. import logger
from ..io import read_raw_fif
from ..io.cli import write_results
from ..utils.list_files import raw_fif_selection
from ..utils._checks import _check_path, _check_n_jobs
from ..utils._docs import fill_doc


mne.set_log_level('ERROR')


@fill_doc
def interpolate_bads(raw):
    """
    Interpolate bad channels.

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    """
    raw.interpolate_bads(reset_bads=True, mode='accurate')
    return raw


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname, dir_in, dir_out):
    """%(pipeline_header)s

    Interpolate bad channels in raw .fif files.

    Parameters
    ----------
    %(fname)s
    %(dir_in)s
    %(dir_out)s

    Returns
    -------
    %(success)s
    %(fname)s
    """
    logger.info('Processing: %s' % fname)
    try:
        # checks paths
        fname = _check_path(fname, item_name='fname', must_exist=True)
        dir_in = _check_path(dir_in, 'dir_in', must_exist=True)
        dir_out = _check_path(dir_out, 'dir_out', must_exist=True)

        # create output file name
        output_fname = _create_output_fname(fname, dir_in, dir_out)

        # preprocess
        raw = read_raw_fif(fname)
        raw = interpolate_bads(raw)

        # export
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname))

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname))


def _create_output_fname(
        fname,
        dir_in,
        dir_out
        ):
    """Creates the output file name based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in dir_in
    relative_fname = fname.relative_to(dir_in)
    # create output fname
    output_fname = dir_out / relative_fname
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname


@fill_doc
def _cli(
        dir_in,
        dir_out,
        n_jobs=1,
        participant=None,
        session=None,
        fname=None,
        ignore_existing: bool = True
        ):
    """%(cli_header)s

    Parameters
    ----------
    %(dir_in)s
    %(dir_out)s
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    %(ignore_existing)s
    """
    # check arguments
    dir_in = _check_path(dir_in, 'dir_in', must_exist=True)
    dir_out = _check_path(dir_out, 'dir_out', must_exist=False)
    n_jobs = _check_n_jobs(n_jobs)

    # create output folder if needed
    os.makedirs(dir_out, exist_ok=True)

    # list files to process
    fifs = raw_fif_selection(
        dir_in,
        dir_out,
        participant=participant,
        session=session,
        fname=fname,
        ignore_existing=ignore_existing
        )

    # create input pool for pipeline
    input_pool = [(fname, dir_in, dir_out) for fname in fifs]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    write_results(results, dir_out / 'bads.pcl')
