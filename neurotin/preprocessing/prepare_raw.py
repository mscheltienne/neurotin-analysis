import multiprocessing as mp
import os
from pathlib import Path
import traceback

import mne

from .bad_channels import PREP_bads_suggestion
from .events import check_events, add_annotations_from_events
from .filters import apply_filter_eeg, apply_filter_aux
from .. import logger
from ..io import read_raw_fif
from ..io.cli import write_results
from ..utils.list_files import raw_fif_selection
from ..utils._checks import _check_path, _check_n_jobs
from ..utils._docs import fill_doc


mne.set_log_level('ERROR')


@fill_doc
def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, add montage, applying FIR filters, and applying a
    common average reference (CAR).

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(raw)s
    %(bads)s
    """
    # Check sampling frequency
    if raw.info['sfreq'] != 512:
        raw.resample(sfreq=512)

    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)
    raw, _ = add_annotations_from_events(raw)

    # Filter
    apply_filter_aux(raw, bandpass=(1., 40.), notch=True)
    apply_filter_eeg(raw, bandpass=(1., 40.))

    # Mark bad channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy and applies notch
    raw.info['bads'] = bads

    # Add montage
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel

    # CAR
    apply_filter_eeg(raw, car=True)

    return raw, bads


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname, dir_in, dir_out):
    """%(pipeline_header)s

    Prepare and preprocess raw .fif files.

    Parameters
    ----------
    %(fname)s
    %(dir_in)s
    %(dir_out)s

    Returns
    -------
    %(success)s
    %(fname)s
    %(bads)s
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
        raw, bads = prepare_raw(raw)
        assert len(raw.info['projs']) == 0  # sanity-check

        # export
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname), bads)

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname), None)


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

    write_results(results, dir_out / 'prepare_raw.pcl')
