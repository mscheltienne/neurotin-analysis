import os
import pickle
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

from events import check_events
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from utils import read_raw_fif, read_exclusion, write_exclusion, list_raw_fif


mne.set_log_level('ERROR')


def prepare_raw(raw):
    """
    Prepare raw instance by checking events, adding events as annotations,
    marking bad channels, adding montage, applying FIR filters, applying CAR
    and interpolating bad channels.

    Parameters
    ----------
    raw : Raw
        Raw instance (will be modified in-place).
    bads : list
        List of interpolated bad channels.

    Returns
    -------
    raw : Raw
        Raw instance (modified in-place).
    """
    # Check events
    recording_type = Path(raw.filenames[0]).stem.split('-')[1]
    check_events(raw, recording_type)

    # Filter AUX
    apply_filter_aux(raw, bandpass=(1., 45.), notch=True)

    # Mark bad channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy
    raw.info['bads'] = bads

    # Reference and filter EEG
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel
    apply_filter_eeg(raw, bandpass=(1., 45.), notch=False, car=True)

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True, mode='accurate')

    return raw, bads


def pipeline(fname, input_dir_fif, output_dir_fif, output_dir_set):
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
    output_dir_set : str | Path
        Path used to save raw in EEGLAB format with the same structure as in
        fname.

    Returns
    -------
    success : bool
        False if a step raised an Exception.
    fname : str
        Path to the input '-raw.fif' file to preprocess.
    bads : list
        List of interpolated bad channels.
    """
    print (f'Preprocessing: {fname}')
    try:
        # checks paths
        fname, output_fname_fif, output_fname_set = \
            _check_paths(fname, input_dir_fif, output_dir_fif, output_dir_set)

        # preprocess
        raw = read_raw_fif(fname)
        raw, bads = prepare_raw(raw)

        # export
        raw.save(output_fname_fif, fmt="double", overwrite=True)
        raw.export(output_fname_set, fmt='eeglab')

        return (True, str(fname), bads)

    except Exception:
        print ('----------------------------------------------')
        print (f'FAILED: {fname} -> Skip.')
        print(traceback.format_exc())
        print ('----------------------------------------------')
        return (False, str(fname), None)


def _check_paths(fname, input_dir_fif, output_dir_fif, output_dir_set):
    """Checks that fname is valid, and create the output_fname_fif and
    output_fname_set from fname and the output_dir."""
    fname = Path(fname)
    input_dir_fif = Path(input_dir_fif)
    output_dir_fif = Path(output_dir_fif)
    output_dir_set = Path(output_dir_set)

    # check existance
    assert fname.exists()
    os.makedirs(output_dir_fif, exist_ok=True)
    os.makedirs(output_dir_set, exist_ok=True)
    # this will fail if fname is not in input_dir_fif
    relative_fname = fname.relative_to(input_dir_fif)

    # create output fname
    output_fname_fif = output_dir_fif / relative_fname
    output_fname_set = output_dir_fif / relative_fname.with_suffix('.set')

    return fname, output_fname_fif, output_fname_set


def main(input_dir_fif, output_dir_fif, output_dir_set, processes=1,
         subject=None, session=None, fname=None):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    input_dir_fif : str | Path
        Path to the folder containing the FIF files to preprocess.
    output_dir_fif : str | Path
        Path to the folder containing the FIF files preprocessed.
    output_dir_set : str | Path
        Path to the folder containing the EEGLAB files preprocessed.
    processes : int
        Number of parallel processes used if semiauto is False.
    subject : int | None
        Restricts file selection to this subject.
    session : int | None
        Restricts file selection to this session.
    fname : str | Path | None
        Restrict file selection to this file (must be inside input_dir_fif).
    """
    input_dir_fif, output_dir_fif, output_dir_set = \
        _check_folders(input_dir_fif, output_dir_fif, output_dir_set)
    processes = _check_processes(processes)
    subject = _check_subject(subject)
    session = _check_session(session)
    fname = _check_fname(fname, input_dir_fif)

    # read excluded files
    exclusion_file = output_dir_fif / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # list files to preprocess
    fifs_in = [f for f in list_raw_fif(input_dir_fif, exclude=exclude)
               if not (output_dir_fif/f.relative_to(input_dir_fif)).exists()]
    subjects = [int(file.parent.parent.parent.name) for file in fifs_in]
    sessions = [int(file.parent.parent.name.split()[1]) for file in fifs_in]

    # filter inputs
    if subject is not None:
        sessions = [session_id for k, session_id in enumerate(sessions)
                    if subjects[k] == subject]
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if subjects[k] == subject]
    if session is not None:
        fifs_in = [file for k, file in enumerate(fifs_in)
                    if sessions[k] == session]
    if fname is not None:
        assert fname in fifs_in
        fifs_in = [fname]

    # create input pool for pipeline based on provided subject info
    input_pool = [(fname, input_dir_fif, output_dir_fif, output_dir_set)
                  for fname in fifs_in]
    assert 0 < len(input_pool)

    with mp.Pool(processes=processes) as p:
        results = p.starmap(pipeline, input_pool)

    with open(output_dir_fif/'bads.pcl', mode='wb') as f:
        pickle.dump(results, f, -1)

    exclude = [file for success, file, _ in results if not success]
    write_exclusion(exclusion_file, exclude)


def _check_folders(input_dir_fif, output_dir_fif, output_dir_set):
    """Checks that the folders exist and are pathlib.Path instances."""
    input_dir_fif = Path(input_dir_fif)
    output_dir_fif = Path(output_dir_fif)
    output_dir_set = Path(output_dir_set)
    assert input_dir_fif.exists(), 'The input folder does not exists.'
    os.makedirs(output_dir_fif, exist_ok=True)
    os.makedirs(output_dir_set, exist_ok=True)
    return input_dir_fif, output_dir_fif, output_dir_set


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
        'input_dir_fif', type=str,
        help='folder containing FIF files to preprocess.')
    parser.add_argument(
        'output_dir_fif', type=str,
        help='folder containing FIF files preprocessed.')
    parser.add_argument(
        'output_dir_set', type=str,
        help='folder containing EEGLAB files preprocessed.')
    parser.add_argument(
        '--processes', type=int, metavar='int',
        help='number of parallel processes.', default=1)
    parser.add_argument(
        '--subject', type=int, metavar='int',
        help='restrict to files with this subject ID.', default=None)
    parser.add_argument(
        '--session', type=int, metavar='int',
        help='restrict with files with this session ID.', default=None)
    parser.add_argument(
        '--fname', type=str, metavar='path',
        help='restrict to this file.', default=None)

    args = parser.parse_args()

    main(args.input_dir_fif, args.output_dir_fif, args.output_dir_set,
         args.processes, args.subject, args.session, args.fname)
