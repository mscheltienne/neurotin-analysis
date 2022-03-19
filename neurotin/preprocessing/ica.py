import multiprocessing as mp
import os
import traceback

import mne

from .. import logger
from ..io.cli import write_results
from ..utils.list_files import raw_fif_selection
from ..utils._checks import _check_path, _check_n_jobs
from ..utils._docs import fill_doc

mne.set_log_level('ERROR')


def _exclude_ocular_components(raw, ica, **kwargs):
    """Find and exclude ocular-related components.
    kwargs are passed to ica.find_bads_eog()."""
    eog_idx, eog_scores = ica.find_bads_eog(raw, **kwargs)
    return eog_idx, eog_scores[eog_idx]


def _exclude_heartbeat_components(raw, ica, **kwargs):
    """Find and exclude heartbeat-related components.
    kwargs are passed to ica.find_bads_ecg()."""
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw, **kwargs)
    return ecg_idx, ecg_scores[ecg_idx]


@fill_doc
def exclude_ocular_and_heartbeat_with_ICA(raw, *, semiauto=False):
    """
    Apply ICA to remove ocular and heartbeat artifacts from raw instance.

    The raw instance is modified in-place.

    Parameters
    ----------
    %(raw)s
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.

    Returns
    -------
    %(raw)s
    ica : ICA
    eog_scores : Scores used for selection of the ocular component(s).
    ecg_scores : Scores used for selection of the heartbeat component(s).
    """
    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    ica.fit(raw, picks=picks)

    raw_ = raw.copy().pick_types(eeg=True, exclude='bads')
    eog_idx, eog_scores = \
        _exclude_ocular_components(raw_, ica, threshold=4.8,
                                   measure='zscore')
    ecg_idx, ecg_scores = \
        _exclude_heartbeat_components(raw_, ica, method='correlation',
                                      threshold=0.7, measure='correlation')

    ica.exclude = eog_idx + ecg_idx

    try:
        assert len(eog_idx) <= 2, 'More than 2 EOG component detected.'
        assert len(ecg_idx) <= 1, 'More than 1 ECG component detected.'
        assert len(ica.exclude) != 0, 'No EOG / ECG component detected.'
    except AssertionError:
        if semiauto:
            ica.plot_scores(eog_scores)
            ica.plot_scores(ecg_scores)
            ica.plot_components(inst=raw_)
            ica.plot_sources(raw_, block=True)
        else:
            raise

    # Apply ICA
    ica.apply(raw)

    # Scores should not be returned when #9846 is fixed.
    return raw, ica, eog_scores, ecg_scores


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname, dir_in, dir_out):
    """%(pipeline_header)s

    Exclude ocular and heartbeat related components with ICA.

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
        output_fname_raw, output_fname_ica = \
            _create_output_fname(fname, dir_in, dir_out)

        # ica
        raw = mne.io.read_raw_fif(fname, preload=True)
        raw, ica, eog_scores, ecg_scores = \
            exclude_ocular_and_heartbeat_with_ICA(raw)

        # export
        raw.save(output_fname_raw, fmt="double", overwrite=True)
        ica.save(output_fname_ica)

        return (True, str(fname), eog_scores, ecg_scores)

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname), None, None)


def _create_output_fname(
        fname,
        dir_in,
        dir_out
        ):
    """Creates the output file names based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(dir_in)
    # create output fname
    output_fname_raw = dir_out / relative_fname
    output_fname_ica = \
        dir_out / str(relative_fname).replace('-raw.fif', '-ica.fif')
    os.makedirs(output_fname_raw.parent, exist_ok=True)
    return output_fname_raw, output_fname_ica


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
    %(input_dir_fif)s
    %(output_dir_fif)s
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
        ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, dir_in, dir_out) for fname in fifs]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    write_results(results, dir_out / 'ica.pcl')
