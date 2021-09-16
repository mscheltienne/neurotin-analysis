import os
import argparse
from pathlib import Path

import mne

from cli import query_yes_no
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from events import add_annotations_from_events, check_events
from utils import read_raw_fif, read_exclusion, write_exclusion, list_raw_fif


mne.set_log_level('ERROR')


def _prepare_raw(fname):
    """
    Pipeline loading file, fixing channel names, fixing channels types,
    checking events, prompting to input interactively bad segments annotations,
    adding events as annotations, prompting to inpt interactively bad channels,
    adding montage, applying FIR filters, applying CAR and interpolating bad
    channels.
    """
    # Load
    raw = read_raw_fif(fname)

    # Check events
    recording_type = Path(fname).stem.split('-')[1]
    check_events(raw, recording_type)

    # Annotate bad segments of data
    raw_ = raw.copy()
    apply_filter_eeg(raw_, bandpass=(1., None), notch=True, car=False)
    apply_filter_aux(raw_, bandpass=(1., None), notch=True)
    raw_.plot(block=True)
    raw.set_annotations(raw_.annotations)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)
    raw_, _ = add_annotations_from_events(raw_)

    # Re-filter for bads marking
    apply_filter_eeg(raw_, bandpass=(1., 40.), notch=True, car=False)
    raw_.set_montage('standard_1020')

    # Mark bad channels
    bads = PREP_bads_suggestion(raw_)
    print ('Suggested bads:', bads)
    raw_.info['bads'] = bads
    raw_.plot_psd(fmin=1, fmax=40, picks='eeg', reject_by_annotation=True)
    raw_.plot(block=True)
    raw.info['bads'] = raw_.info['bads']

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
    _, eog_scores = ica.find_bads_eog(raw)
    _, ecg_scores = ica.find_bads_ecg(raw)
    ica.plot_scores(eog_scores)
    ica.plot_scores(ecg_scores)
    ica.plot_sources(raw, block=True)
    # ica.plot_components(inst=raw)
    assert len(ica.exclude) != 0
    ica.apply(raw)

    raw.info['bads'] = bads # bug fixed in #9719
    return raw


def _add_subject_info(raw, subject, birthday, sex):
    """Add subject information to raw instance."""
    raw.info['subject_info'] = dict()
    # subject ID
    raw.info['subject_info']['id'] = subject
    raw.info['subject_info']['his_id'] = str(subject).zfill(3)
    # subject birthday (year, month, day)
    if birthday is not None:
        raw.info['subject_info']['birthday'] = birthday
    # subject sex - (0, 1, 2) for (Unknown, Male, Female)
    raw.info['subject_info']['sex'] = sex

    return raw


def main(subject, birthday, sex, folder_in, folder_out):
    """
    Main preprocessing pipeline, called once per participant.

    Parameters
    ----------
    subject : int
        ID of the subject.
    birthday : 3-length tuple of int (year, month, day)
        Birthday of the subject.
    sex : int
        Sex of the subject. 1: Male - 2: Female.
    folder_in : str | Path
        Path to the folder containing the FIF files to preprocess.
    folder_out : str | Path
        Path to the folder containing the FIF files preprocessed.
    """
    subject_folder = str(subject).zfill(3)
    dirname_in = folder_in / subject_folder
    dirname_out = folder_out / subject_folder
    exclusion_file = folder_out / 'exclusion.txt'
    assert dirname_in.exists()
    os.makedirs(dirname_out, exist_ok=True)
    exclude = read_exclusion(exclusion_file)

    for fif_in in list_raw_fif(dirname_in, exclude=exclude):
        fif_out = dirname_out / fif_in.relative_to(dirname_in)
        if fif_out.exists():
            print(f"Already preprocessed {fif_in.relative_to(dirname_in)}")
            continue
        print(f"Preprocessing {fif_in.relative_to(dirname_in)}")

        try:
            raw = _prepare_raw(fif_in)
            raw = _exclude_EOG_ECG_with_ICA(raw)
            raw = _add_subject_info(raw, subject, birthday, sex)
            raw.info._check_consistency()
            os.makedirs(fif_out.parent, exist_ok=True)
            raw.save(fif_out, fmt="double")

        except AssertionError:
            exclude.append(fif_in)
            write_exclusion(exclusion_file, fif_in)

        if not query_yes_no('Continue?'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NeuroTin preprocessing pipeline',
        description='Preprocess NeuroTin raw FIF files semi-automatically.')
    parser.add_argument(
        'subject', type=int, metavar='int',
        help='ID of the subject.')
    parser.add_argument(
        'birthday', type=str,
        help='Birthday of the subject as a tuple of int (year, month, day)')
    parser.add_argument(
        'sex', type=int, metavar='int',
        help='Sex of the subject.')
    parser.add_argument(
        'folder_in', type=str,
        help='Folder containing FIF files to preprocess.')
    parser.add_argument(
        'folder_out', type=str,
        help='Folder containing FIF files preprocessed.')
    args = parser.parse_args()

    main(args.subject, args.birthday, args.sex, args.folder_in,
         args.folder_out)
