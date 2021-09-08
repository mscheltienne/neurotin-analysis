import os
from pathlib import Path

import mne

from cli import input_participant
from bad_channels import PREP_bads_suggestion
from filters import apply_filter_eeg, apply_filter_aux
from events import add_annotations_from_events, check_events
from utils import read_raw_fif, read_exclusion, write_exclusion, list_raw_fif


FOLDER_IN = Path(r"/Users/scheltie/Documents/NeuroTin Data/Raw/")
FOLDER_OUT = Path(r"/Users/scheltie/Documents/NeuroTin Data/Clean/")


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

    # Annotate bad segments of data
    raw_ = raw.copy()
    apply_filter_eeg(raw_, bandpass=(1., None), notch=True, car=False)
    apply_filter_aux(raw_, bandpass=(1., None), notch=True)
    raw_.plot(block=True)
    raw.set_annotations(raw_.annotations)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)
    raw_, _ = add_annotations_from_events(raw_)

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
    apply_filter_eeg(raw_, bandpass=(1., 40.), notch=False, car=True)
    apply_filter_aux(raw_, bandpass=(1., 40.), notch=True)

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
    # Reset bads, bug described in #9716
    bads = raw.info['bads']
    raw.info['bads'] = list()

    ica = mne.preprocessing.ICA(method='picard')
    ica.fit(raw, picks='eeg', reject_by_annotation=True)
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica.plot_scores(eog_scores)
    ica.plot_scores(ecg_scores)
    ica.plot_sources(raw, block=True)
    # ica.plot_components(inst=raw)
    assert len(ica.exclude) != 0
    ica.apply(raw)

    raw.info['bads'] = bads # bug described in #9716
    return raw


def main():
    """
    Main preprocessing pipeline, called once per participant.
    """
    _, participant_folder = input_participant(FOLDER_IN)
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
        print(f"Preprocesing {fif_in.relative_to(dirname_in)}")
        try:
            raw = preprocessing_pipeline(fif_in)
            raw = ICA_pipeline(raw)
        except AssertionError:
            exclude.append(fif_out)
            write_exclusion(exclusion_file, fif_out)
        raw.save(fif_out, fmt="double")
