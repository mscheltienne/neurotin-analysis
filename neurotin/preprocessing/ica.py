import os
import traceback
import multiprocessing as mp

import mne

from .. import logger
from ..io.cli_results import write_results
from ..io.list_files import raw_fif_selection
from ..utils.docs import fill_doc
from ..utils.checks import _check_path, _check_n_jobs

mne.set_log_level('ERROR')


def _ica(raw, **kwargs):
    """Fit an ICA with the given kwargs to raw EEG channels.
    kwargs are passed to mne.preprocessing.ICA()."""
    ica = mne.preprocessing.ICA(**kwargs)
    ica.fit(raw, picks='eeg')
    return ica


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

    Parameters
    ----------
    %(raw_in_place)s
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.

    Returns
    -------
    %(raw_in_place)s
    ica : ICA instance.
    eog_scores : Scores used for selection of the ocular component(s).
    ecg_scores : Scores used for selection of the heartbeat component(s).
    """
    ica = _ica(raw, method='picard', max_iter='auto')

    eog_idx, eog_scores = \
        _exclude_ocular_components(raw, ica, threshold=0.6,
                                   measure='correlation')
    ecg_idx, ecg_scores = \
        _exclude_heartbeat_components(raw, ica, method='correlation',
                                      threshold=6.6, measure='zscore')

    ica.exclude = eog_idx + ecg_idx

    try:
        assert len(eog_idx) <= 2, 'More than 2 EOG component detected.'
        assert len(ecg_idx) <= 1, 'More than 1 ECG component detected.'
        assert len(ica.exclude) != 0, 'No EOG/ECG component detected.'
    except AssertionError:
        if semiauto:
            ica.plot_scores(eog_scores)
            ica.plot_scores(ecg_scores)
            ica.plot_sources(raw, block=True)
        else:
            raise

    # Apply ICA
    ica.apply(raw)

    # Scores should not be returned when #9846 is fixed.
    return raw, ica, eog_scores, ecg_scores


# -----------------------------------------------------------------------------
@fill_doc
def _pipeline(fname, input_dir_fif, output_dir_fif):
    """%(pipeline_header)s

    Exclude ocular and heartbeat related components with ICA.

    Parameters
    ----------
    %(fname)s
    %(input_dir_fif)s
    %(output_dir_fif)s

    Returns
    -------
    %(success)s
    %(fname)s
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
        output_fname, ica_output_fname = \
            _create_output_fname(fname, input_dir_fif, output_dir_fif)

        # ica
        raw = mne.io.read_raw_fif(fname, preload=True)
        raw, ica, eog_scores, ecg_scores = \
            exclude_ocular_and_heartbeat_with_ICA(raw)

        # export
        raw.save(output_fname, fmt="double", overwrite=True)
        ica.save(ica_output_fname)

        return (True, str(fname), eog_scores, ecg_scores)

    except Exception:
        logger.warning('FAILED: %s -> Skip.' % fname)
        logger.debug(traceback.format_exc())
        return (False, str(fname), None, None)


def _create_output_fname(fname, input_dir_fif, output_dir_fif):
    """Creates the output file names based on the relative path between fname
    and input_dir_fif."""
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(input_dir_fif)
    # create output fname
    output_fname = output_dir_fif / relative_fname
    ica_output_fname = \
        output_dir_fif / str(relative_fname).replace('-raw.fif', '-ica.fif')
    os.makedirs(output_fname.parent, exist_ok=True)
    return output_fname, ica_output_fname


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

    # list files to process
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif,
                                participant=participant, session=session,
                                fname=fname, ignore_existing=ignore_existing)

    # create input pool for pipeline
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(_pipeline, input_pool)

    write_results(results, output_dir_fif/'ica.pcl')
