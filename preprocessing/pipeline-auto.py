import os
from pathlib import Path

import mne

from cli import input_participant, input_sex
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from events import add_annotations_from_events, check_events
from utils import read_raw_fif, read_exclusion, write_exclusion, list_raw_fif


mne.set_log_level('WARNING')

FOLDER_IN = Path(r"/Users/scheltie/Documents/NeuroTin Data/Raw/")
FOLDER_OUT = Path(r"/Users/scheltie/Documents/NeuroTin Data/Clean-Auto/")


def preprocessing_pipeline(fname):
    """
    Preprocessing pipeline to annotate bad segments of data, to annotate bad
    channels, to annotate events, to add the reference and the montage, to
    clean the data and to interpolate the bad channels.

    Parameters
    ----------
    fname : str | Path
        Path to the '-raw.fif' file to preprocess.

    Returns
    -------
    raw : Raw
        Raw instance.
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


def ICA_pipeline(raw):
    """
    Apply ICA to remove EOG and ECG artifacts.

    Parameters
    ----------
    raw : Raw
        Raw instance to modify.

    Returns
    -------
    raw : Raw
        Raw instance modified in-place.
    """
    # Reset bads, bug fixed in #9719
    bads = raw.info['bads']
    raw.info['bads'] = list()

    ica = mne.preprocessing.ICA(method='picard')
    ica.fit(raw, picks='eeg', reject_by_annotation=True)
    eog_idx, eog_scores = ica.find_bads_eog(raw)
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw)
    ica.exclude = eog_idx + ecg_idx

    assert len(ica.exclude) != 0
    raw.info['bads'] = bads # bug fixed in #9719
    return raw


def main():
    """
    Main preprocessing pipeline.
    """
    _, participant_folder = input_participant(FOLDER_IN)
    sex = input_sex()
    dirname_in = FOLDER_IN / participant_folder
    dirname_out = FOLDER_OUT / participant_folder
    exclusion_file = FOLDER_OUT / 'exclusion.txt'
    assert dirname_in.exists()
    os.makedirs(dirname_out, exist_ok=True)
    exclude = read_exclusion(exclusion_file)

    fifs = list_raw_fif(dirname_in)
    for fif_in in fifs:
        fif_out = dirname_out / fif_in.relative_to(dirname_in)
        if fif_out.exists() or fif_out in exclude:
            continue
        os.makedirs(fif_out.parent, exist_ok=True)
        print("-------------------------------------------------------------")
        print(f"Preprocessing {fif_in.relative_to(dirname_in)}")
        try:
            raw = preprocessing_pipeline(fif_in)
            raw = ICA_pipeline(raw)
            raw.info['subject_info']['sex'] = sex
            raw.info._check_consistency()
        except AssertionError:
            exclude.append(fif_out)
            write_exclusion(exclusion_file, fif_out)
        raw.save(fif_out, fmt="double")
