from utils import read_raw_fif
from filters import apply_filter
from annotations import add_annotations_from_events


def preprocessing_pipeline(fname):
    # Load
    raw = read_raw_fif(fname)

    # Annotate bad segments of data
    raw_ = apply_filter(
        raw.copy(), car=False, bandpass=(1., None),
        notch=['eeg', 'eog', 'ecg'])
    raw_.plot(block=True)
    raw.set_annotations(raw_.annotations)

    # Add event annotations
    raw, _ = add_annotations_from_events(raw)

    # Mark bad channels
    raw_.plot_psd(fmin=1, fmax=40, picks='eeg', reject_by_annotation=True)
    raw_.plot(block=True)
    raw.info['bads'] = raw_.info['bads']

    # Reference and filter
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')
    raw = apply_filter(raw, car=True, bandpass=(1., 40.), notch=['eog', 'ecg'])

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=False, mode='accurate', method='spline')

    return raw
