import os
import pickle
import argparse
import traceback
import multiprocessing as mp

import mne

from utils.list_files import raw_fif_selection
from utils.checks import _check_path, _check_n_jobs
from utils.exclusion import read_exclusion, write_exclusion

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


def exclude_ocular_and_heartbeat_with_ICA(raw, *, semiauto=False):
    """
    Apply ICA to remove ocular and heartbeat artifacts from raw instance.

    Parameters
    ----------
    raw : raw : Raw
        Raw instance modified in-place.
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.

    Returns
    -------
    raw : Raw instance modified in-place.
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


def pipeline(fname, input_dir_fif, output_dir_fif):
    """
    Pipeline function called on each raw file.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    input_dir_fif : str | Path
        Path to the input raw directory (parent from fname).
    output_dir_fif : str | Path
        Path used to save raw in MNE format with the same structure as in
        fname.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to preprocess.
    """
    print ('Preprocessing: %s' % fname)
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
        print ('----------------------------------------------')
        print ('FAILED: %s -> Skip.' % fname)
        print(traceback.format_exc())
        print ('----------------------------------------------')
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


def main(input_dir_fif, output_dir_fif, n_jobs=1, subject=None, session=None,
         fname=None, ignore_existing=True):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files preprocessed on which ICA
        must be applied.
    output_dir_fif : str | Path
        Path to the folder containing the FIF files without ocular and
        heartbeat components.
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
        If True, files already preprocessed are not included.
    """
    # check arguments
    input_dir_fif = _check_path(input_dir_fif, item_name='input_dir_fif',
                                must_exist=True)
    output_dir_fif = _check_path(output_dir_fif, item_name='output_dir_fif')
    n_jobs = _check_n_jobs(n_jobs)

    # create output folder if needed
    os.makedirs(output_dir_fif, exist_ok=True)

    # read excluded files
    exclusion_file = output_dir_fif / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # list files to preprocess
    fifs_in = raw_fif_selection(input_dir_fif, output_dir_fif, exclude,
                                subject=subject, session=session, fname=fname)

    # create input pool for pipeline based on provided subject info
    input_pool = [(fname, input_dir_fif, output_dir_fif)
                  for fname in fifs_in]
    assert 0 < len(input_pool)  # sanity-check

    with mp.Pool(processes=n_jobs) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_fif/'ica.pcl', mode='wb') as f:
        pickle.dump([(file, eog_scores, ecg_scores)
                     for success, file, eog_scores, ecg_scores in results
                     if success], f, -1)

    exclude = [file for success, file, _, _ in results if not success]
    write_exclusion(exclusion_file, exclude)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin ICA preprocessing pipeline.',
        description='Apply ICA on NeuroTin preprocess raw FIF files.')
    parser.add_argument(
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_fif', type=str,
        help='folder containing FIF files preprocessed.')
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

    main(args.input_dir_fif, args.output_dir_fif, args.n_jobs, args.subject,
         args.session, args.fname, args.ignore_existing)
