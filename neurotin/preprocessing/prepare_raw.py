import os
import pickle
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

from .. import logger
from .bad_channels import PREP_bads_suggestion
from .filters import apply_filter_eeg, apply_filter_aux
from .events import check_events, add_annotations_from_events
from ..io import read_raw_fif
from ..io.list_files import raw_fif_selection, read_exclusion, write_exclusion
from ..utils.docs import fill_doc
from ..utils.checks import _check_path, _check_n_jobs


mne.set_log_level('ERROR')


@fill_doc
def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, adding montage, applying FIR filters, applying CAR
    and interpolating bad channels.

    Parameters
    ----------
    %(raw_in_place)s

    Returns
    -------
    %(raw_in_place)s
    %(bads)s
    """
    # Check sampling frequency
    if raw.info['sfreq'] != 512:
        raw.resample(sfreq=512)

    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)  # raise if there is a problem
    raw, _ = add_annotations_from_events(raw)

    # Filter AUX
    apply_filter_aux(raw, bandpass=(1., 45.), notch=True)

    # Mark bad channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy
    raw.info['bads'] = bads

    # Reference (projector) and filter EEG
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel
    apply_filter_eeg(raw, bandpass=(1., 45.), notch=False, car=True)

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True, mode='accurate')

    return raw, bads


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname, input_dir_fif, output_dir_fif):
    """%(pipeline_header)s

    Prepare and preprocess raw .fif files.

    Parameters
    ----------
    %(fname)s
    %(input_dir_fif)s
    %(output_dir_fif)s

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
        input_dir_fif = _check_path(input_dir_fif,
                                    item_name='input_dir_fif',
                                    must_exist=True)
        output_dir_fif = _check_path(output_dir_fif,
                                     item_name='output_dir_fif',
                                     must_exist=True)

        # create output file name
        output_fname = _create_output_fname(fname, input_dir_fif,
                                            output_dir_fif)

        # preprocess
        raw = read_raw_fif(fname)
        raw, bads = prepare_raw(raw)
        raw.apply_proj()

        # export
        raw.save(output_fname, fmt="double", overwrite=True)

        return (True, str(fname), bads)

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname), None)


def _create_output_fname(fname, input_dir_fif, output_dir_fif):
    """Creates the output file name based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(input_dir_fif)
    # create output fname
    output_fname = output_dir_fif / relative_fname
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname


@fill_doc
def _cli(input_dir_fif, output_dir_fif, n_jobs=1, participant=None,
         session=None, fname=None, ignore_existing=True):
    """%(cli_header)s

    Parameters
    ----------
    %(input_dir_fif)s
    %(output_dir_fif)s
    %(n_jobs)s
    %(select_participant)s
    %(select_session)s
    %(select_fname)s
    %(ignore_existing)s
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, item_name='input_dir_fif',
                                must_exist=True)
    output_dir_fif = _check_path(output_dir_fif, item_name='output_dir_fif')
    n_jobs = _check_n_jobs(n_jobs)

    # create output folder if needed
    os.makedirs(output_dir_fif, exist_ok=True)

    # read exclusion file (create if needed)
    exclusion_file = output_dir_fif / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude,
                                participant=participant, session=session,
                                fname=fname, ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    with open(output_dir_fif/'bads.pcl', mode='wb') as f:
        pickle.dump([(file, bads) for success, file, bads in results
                     if success], f, -1)

    exclude = [file for success, file, _ in results if not success]
    write_exclusion(exclusion_file, exclude)
