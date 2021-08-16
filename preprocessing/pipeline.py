from utils import read_raw_fif
from filters import apply_filter


def preprocessing_pipeline(fname):
    raw = read_raw_fif(fname)
    apply_filter(raw, car=True, bandpass=(1., None), notch=True)


if __name__ == '__main__':
    fname = r'/Volumes/NeuroTin-EEG/Data/Participants/061/Session 14/Calibration/1-calibration-eegoSports 000479-raw.fif'
    preprocessing_pipeline(fname)
