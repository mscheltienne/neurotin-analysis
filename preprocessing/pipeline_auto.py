import os
import argparse
import traceback
from pathlib import Path
import multiprocessing as mp

import mne

from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from events import add_annotations_from_events, check_events
from utils import read_raw_fif, read_exclusion, write_exclusion, list_raw_fif


mne.set_log_level('ERROR')


def _prepare_raw(fname):
    """
    Pipeline loading file, fixing channel names, fixing channels types,
    checking events, adding events as annotations, marking bad channels, adding
    montage, applying FIR filters, applying CAR and interpolating bad channels.
    """
    # Load
    raw = read_raw_fif(fname)

    # Check events
    recording_type = Path(fname).stem.split('-')[1]
    check_events(raw, recording_type)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)

    # Mark bad channels
    raw_ = raw.copy()
    apply_filter_eeg(raw_, bandpass=(1., 40.), notch=True, car=False)
    apply_filter_aux(raw_, bandpass=(1., 40.), notch=True)
    bads = PREP_bads_suggestion(raw_)
    raw.info['bads'] = bads

    # Reference and filter
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')
    apply_filter_eeg(raw, bandpass=(1., 40.), notch=False, car=True)
    apply_filter_aux(raw, bandpass=(1., 40.), notch=True)

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=False, mode='accurate')

    return raw


def _exclude_EOG_ECG_with_ICA(raw):
    """
    Apply ICA to remove EOG and ECG artifacts.
    """
    # Reset bads, bug fixed in #9719
    bads = raw.info['bads']
    raw.info['bads'] = list()

    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
    ica.fit(raw, picks='eeg', reject_by_annotation=True)
    eog_idx, eog_scores = ica.find_bads_eog(raw)
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw)
    ica.exclude = eog_idx + ecg_idx
    assert len(ica.exclude) != 0
    ica.apply(raw)

    raw.info['bads'] = bads # bug fixed in #9719
    return raw


def _add_subject_info(raw, subject, birthyear, sex):
    """Add subject information to raw instance."""
    raw.info['subject_info'] = dict()
    # subject ID
    subject = _check_subject(subject)
    raw.info['subject_info']['id'] = subject
    raw.info['subject_info']['his_id'] = str(subject).zfill(3)
    # subject birthyear
    raw.info['subject_info']['birthyear'] = _check_birthyear(birthyear)
    # subject sex - (0, 1, 2) for (Unknown, Male, Female)
    raw.info['subject_info']['sex'] = _check_sex(sex)

    return raw


def _check_subject(subject):
    """Checks that the subject ID is valid."""
    try:
        subject = int(subject)
    except:
        subject = None
    return subject


def _check_birthyear(birthyear):
    """Checks that the birthyear format is valid."""
    try:
        birthyear = int(birthyear)
        assert 1900 <= birthyear <= 2020
    except:
        birthyear = None
    return birthyear


def _check_sex(sex):
    """Checks that sex is either 1 for Male or 2 for Female. Else returns 0 for
    unknown."""
    try:
        sex = int(sex)
        assert sex in (1, 2)
    except (ValueError, TypeError, AssertionError):
        sex = 0
    return sex


def pipeline(fname, fname_out, subject, birthyear, sex):
    """
    Pipeline function called by each process for each file.

    Parameters
    ----------
    fname : str | Path
        Path to the input '-raw.fif' file to preprocess.
    fname_out : str | Path
        Path to the output '-raw.fif' file preprocessed.
    subject : int
        ID of the subject.
    birthyear : int
        Birthyear of the subject.
    sex : int
        Sex of the subject. 1: Male - 2: Female.

    Returns
    -------
    success : bool
        False if a step raised an AssertionError.
    fname : Path
        Path to the input '-raw.fif' file to preprocess.
    """
    print (f'Preprocessing: {fname}')
    try:
        # Preprocess
        raw = _prepare_raw(_check_fname(fname))
        raw = _exclude_EOG_ECG_with_ICA(raw)
        raw = _add_subject_info(raw, subject, birthyear, sex)
        raw.info._check_consistency()
        # Export
        raw.save(_check_fname_out(fname_out), fmt="double", overwrite=False)
        return (True, fname)

    except Exception:
        print (f'FAILED: {fname}')
        print(traceback.format_exc())
        return (False, fname)


def _check_fname(fname):
    """Checks that the input file exists."""
    fname = Path(fname)
    assert fname.exists()
    return fname


def _check_fname_out(fname_out):
    """Checks that fname_out is a Path and create needed directories."""
    fname_out = Path(fname_out)
    os.makedirs(fname_out.parent, exist_ok=True)
    return fname_out


def main(subject_info, folder_in, folder_out, processes=1):
    """
    Main preprocessing pipeline.

    Parameters
    ----------
    subject_info : str | Path
        Path to the file containing the subject information to parse.
    folder_in : str | Path
        Path to the folder containing the FIF files to preprocess.
    folder_out : str | Path
        Path to the folder containing the FIF files preprocessed.
    processes : int
        Number of parallel processes.
    """
    subject_info = _parse_subject_info(subject_info)
    folder_in, folder_out = _check_folders(folder_in, folder_out)
    processes = _check_processes(processes)

    exclusion_file = folder_out / 'exclusion.txt'
    exclude = read_exclusion(exclusion_file)

    # List files to preprocess
    fifs_in = [fname for fname in list_raw_fif(folder_in, exclude=exclude)
               if not (folder_out / fname.relative_to(folder_in)).exists()]

    # create input pool for pipeline based on provided subject info
    subjects = [int(fname.parent.parent.parent.name) for fname in fifs_in]
    input_pool = [(fifs_in[k], folder_out / fifs_in[k].relative_to(folder_in),
                   idx, subject_info[idx][0], subject_info[idx][1])
                  for k, idx in enumerate(subjects) if idx in subject_info]

    with mp.Pool(processes=processes) as p:
        results = p.starmap(pipeline, input_pool)

    exclude = [fname for success, fname in results if not success]
    write_exclusion(exclusion_file, exclude)


def _parse_subject_info(subject_info):
    """
    Parse the subject_info file and return the subject ID with his birthyear
    and sex.

    Returns
    -------
    dict
        key : int
            ID of the subject.
        value : 2-length tuple (birtday, sex)
            birthyear : int
                Birthyear of the subject.
            sex : int
                Sex of the subject. 1: Male - 2: Female."""
    subject_info = Path(subject_info)
    assert subject_info.exists()
    with open(subject_info, 'r') as file:
        lines = file.readlines()
    lines = [line.strip().split() for line in lines if len(line) > 0]
    lines = [[eval(l.strip()) for l in line]
             for line in lines if len(line) == 3]
    return {line[0]: (line[1], line[2]) for line in lines}


def _check_folders(folder_in, folder_out):
    """Checks that the folder exists and are pathlib.Path instances."""
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)
    assert folder_in.exists(), 'The input folder does not exists.'
    os.makedirs(folder_out, exist_ok=True)
    return folder_in, folder_out


def _check_processes(processes):
    """Checks that the number of processes is valid."""
    processes = int(processes)
    assert 0 < processes
    return processes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin preprocessing pipeline',
        description='Preprocess NeuroTin raw FIF files.')
    parser.add_argument(
        'subject_info', type=str,
        help='File containing the subject information to parse.')
    parser.add_argument(
        'folder_in', type=str,
        help='Folder containing FIF files to preprocess.')
    parser.add_argument(
        'folder_out', type=str,
        help='Folder containing FIF files preprocessed.')
    parser.add_argument(
        '-p', '--processes', type=int, metavar='int',
        help='Number of parallel processes.', default=1)
    args = parser.parse_args()

    main(args.subject_info, args.folder_in, args.folder_out, args.processes)
