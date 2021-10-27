import os
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

from meas_info import fill_info
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from events import add_annotations_from_events, check_events
from utils import (read_raw_fif, read_exclusion, write_exclusion, list_raw_fif,
                   parse_subject_info)


mne.set_log_level('ERROR')


def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, adding montage, applying FIR filters, applying CAR
    and interpolating bad channels.

    Parameters
    ----------
    raw : Raw
        Raw instance modified in-place.

    Returns
    -------
    raw : Raw instance modified in-place.
    """
    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)

    # Filter AUX
    apply_filter_aux(raw, bandpass=(1., 40.), notch=True)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)

    # Mark bad channels
    raw.info['bads'] = PREP_bads_suggestion(raw)  # operates on a copy

    # Reference and filter EEG
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel
    apply_filter_eeg(raw, bandpass=(1., 40.), notch=False, car=True)

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=False, mode='accurate')

    return raw


def exclude_EOG_ECG_with_ICA(raw, semiauto=False):
    """
    Apply ICA to remove EOG and ECG artifacts from raw instance.

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
    eog_scores : Scores used for selection of the EOG component(s).
    ecg_scores : Scores used for selection of the ECG component(s).
    """
    # Reset bads, bug fixed in #9719
    bads = raw.info['bads']
    raw.info['bads'] = list()

    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
    ica.fit(raw, picks='eeg')
    eog_idx, eog_scores = ica.find_bads_eog(raw, threshold=0.6,
                                            measure='correlation')
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method='correlation',
                                            threshold=6.8, measure='zscore')
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
    ica.apply(raw)

    raw.info['bads'] = bads # bug fixed in #9719

    # To be rework when #9846 is fixed.
    return raw, ica, eog_scores[eog_idx], ecg_scores[ecg_idx]


def pipeline(fname, fname_out_stem, semiauto, subject, sex, birthday):
    """
    Pipeline function called on each raw file.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    fname_out_stem : str | Path
        Path and naming scheme used to save -raw.fif and -ica.fif files.
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.
    subject : int
        ID of the subject.
    sex : int
        Sex of the subject. 1: Male - 2: Female.
    birthday : 3-length tuple of int
        Subject's birthday as (year, month, day).

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to preprocess.
    eog_scores : array (n, )
        Scores used for selection of the EOG component(s).
    ecg_scores : array (n, )
        Scores used for selection of the ECG component(s).
    """
    # To be rework when #9846 is fixed.
    print (f'Preprocessing: {fname}')
    try:
        # Preprocess
        raw = read_raw_fif(fname)
        raw = fill_info(raw, subject, sex, birthday)
        raw = prepare_raw(raw)
        raw, ica, eog_scores, ecg_scores = exclude_EOG_ECG_with_ICA(
            raw, semiauto=semiauto)
        # Export
        raw.save(_check_fname_out_stem(fname_out_stem, 'raw'),
                 fmt="double", overwrite=False)
        ica.save(_check_fname_out_stem(fname_out_stem, 'ica'))
        return (True, str(fname), eog_scores, ecg_scores)

    except Exception:
        print (f'FAILED: {fname}')
        print(traceback.format_exc())
        return (False, str(fname), None, None)


def _check_fname_out_stem(fname_out_stem, type_):
    """Checks that fname_out_stem is a valid, appends the correct extension and
    create needed directories."""
    type_ = type_.strip().lower()
    assert type_ in ['raw', 'ica']
    if type_ == 'raw':
        fname_out = Path(str(fname_out_stem) + '-raw.fif')
    elif type_ == 'ica':
        fname_out = Path(str(fname_out_stem) + '-ica.fif')
    os.makedirs(fname_out.parent, exist_ok=True)
    return fname_out


def main(folder_in, folder_out, subject_info_fname, semiauto=False,
         processes=1, subject=None, session=None, fname=None):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    folder_in : str | Path
        Path to the folder containing the FIF files to preprocess.
    folder_out : str | Path
        Path to the folder containing the FIF files preprocessed.
    subject_info_fname : str | Path
        Path to the subject info file.
    semiauto : bool
        If True, the user will interactively set annotations, mark bad channels
        and exclude ICA components.
    processes : int
        Number of parallel processes used if semiauto is False.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside folder_in).
    """
    # Checks
    subject_info = parse_subject_info(subject_info_fname)
    folder_in, folder_out = _check_folders(folder_in, folder_out)
    processes = _check_processes(processes)
    subject = _check_subject(subject)
    session = _check_session(session)
    fname = _check_fname(fname, folder_in)

    # Read excluded files
    exclusion_file = folder_out / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # List files to preprocess
    fifs_in = [file for file in list_raw_fif(folder_in, exclude=exclude)
               if not (folder_out / file.relative_to(folder_in)).exists()]
    subjects = [int(file.parent.parent.parent.name) for file in fifs_in]
    sessions = [int(file.parent.parent.name.split()[1]) for file in fifs_in]
    if subject is not None:
        sessions = [session_id for k, session_id in enumerate(sessions)
                    if subjects[k] == subject]
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if subjects[k] == subject]
        subjects = [subject_id for subject_id in subjects
                    if subject_id == subject]
    if session is not None:
        subjects = [subject_id for k, subject_id in enumerate(subjects)
                    if sessions[k] == session]
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if sessions[k] == session]
        sessions = [session_id for session_id in sessions
                    if session_id == session]
    if fname is not None:
        assert fname in fifs_in
        subjects = [subjects[fifs_in.index(fname)]]
        sessions = [sessions[fifs_in.index(fname)]]
        fifs_in = [fname]

    # Create input pool for pipeline based on provided subject info
    input_pool = [(fifs_in[k],
                   str(folder_out / fifs_in[k].relative_to(folder_in))[:-8],
                   semiauto,
                   idx,
                   subject_info[idx][0],
                   subject_info[idx][1])
                  for k, idx in enumerate(subjects) if idx in subject_info]
    assert 0 < len(input_pool)

    with mp.Pool(processes=processes) as p:
        results = p.starmap(pipeline, input_pool)

    # To be rework when #9846 is fixed.
    with open(folder_out / 'data-ica.pcl', mode='wb') as f:
        pickle.dump(results, f, -1)

    exclude = [file for success, file, _, _ in results if not success]
    write_exclusion(exclusion_file, exclude)


def _check_folders(folder_in, folder_out):
    """Checks that the folders exist and are pathlib.Path instances."""
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)
    assert folder_in.exists(), 'The input folder does not exists.'
    os.makedirs(folder_out, exist_ok=True)
    return folder_in, folder_out


def _check_processes(processes):
    """Checks that the number of processes is valid."""
    processes = int(processes)
    assert 0 < processes, 'processes should be a positive integer'
    return processes


def _check_subject(subject):
    """Checks that the subject ID is valid."""
    if subject is not None:
        subject = int(subject)
        assert 0 < subject, 'subject should be a positive integer'
    return subject


def _check_session(session):
    """Checks that the session ID is valid."""
    if session is not None:
        session = int(session)
        assert 1 <= session <= 15, 'session should be included in (1, 15)'
    return session


def _check_fname(fname, folder_in):
    """Checks that the fname is valid."""
    if fname is not None:
        fname = Path(fname)
        try:
            fname.relative_to(folder_in)
        except ValueError:
            raise AssertionError('fname not in folder_in')
    return fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin preprocessing pipeline.',
        description='Preprocess NeuroTin raw FIF files.')
    parser.add_argument(
        'folder_in', type=str,
        help='Folder containing FIF files to preprocess.')
    parser.add_argument(
        'folder_out', type=str,
        help='Folder containing FIF files preprocessed.')
    parser.add_argument(
        'subject_info_fname', type=str,
        help='File containing the subject information to parse.')
    parser.add_argument(
        '--semiauto', dest='semiauto', action='store_true',
        help='Flag to enable semi-automatic preprocessing.')
    parser.add_argument(
        '--processes', type=int, metavar='int',
        help='Number of parallel processes (if auto).', default=1)
    parser.add_argument(
        '--subject', type=int, metavar='int',
        help='Restrict to files with this subject ID.', default=None)
    parser.add_argument(
        '--session', type=int, metavar='int',
        help='Restrict with files with this session ID.', default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='Restrict to this file.', default=None)

    args = parser.parse_args()

    main(args.folder_in, args.folder_out, args.subject_info_fname,
         args.semiauto, args.processes, args.subject, args.session, args.fname)
