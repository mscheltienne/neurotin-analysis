from pathlib import Path

from mne.preprocessing import ICA

from utils import read_raw_fif
from filters import apply_filter
from bad_channels import RANSAC_bads_suggestion
from events import add_annotations_from_events, check_events


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
    raw : Raw instance.
    """
    # Load
    raw = read_raw_fif(fname)

    # Check events
    recording_type = Path(fname).stem.split('-')[1]
    check_events(raw, recording_type)

    # Annotate bad segments of data
    raw_ = apply_filter(
        raw.copy(), car=False, bandpass=(1., None),
        notch=['eeg', 'eog', 'ecg'])
    raw_.plot(block=True)
    raw.set_annotations(raw_.annotations)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)
    raw_, _ = add_annotations_from_events(raw_)

    # Mark bad channels
    bads = RANSAC_bads_suggestion(raw_)
    print ('Suggested bads:', bads)
    raw_.plot_psd(fmin=1, fmax=40, picks='eeg', reject_by_annotation=True)
    raw_.plot(block=True)
    raw.info['bads'] = raw_.info['bads']

    # Reference and filter
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')
    raw = apply_filter(raw, car=True, bandpass=(1., 40.), notch=['eog', 'ecg'])

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=False, mode='accurate')

    # Apply ICA
    ica = ICA(n_components=0.99, max_iter='auto')
    ica.fit(raw)
    # ica.plot_sources(raw)
    # ica.plot_components(inst=raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica.plot_scores(eog_scores)
    ica.plot_scores(ecg_scores)

    ica.exclude = eog_indices
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    return raw
