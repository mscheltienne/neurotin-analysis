import mne

from .. import logger
from ..utils._docs import fill_doc


@fill_doc
def read_raw_fif(fname):
    """
    Load a RAW instance from a .fif file. Renames the EEG channels to match the
    standard 10/20 convention and rename the AUX channels to ECG and EOG.

    Parameters
    ----------
    fname : path-like
        Path to the MNE raw file to read in .fif format.

    Returns
    -------
    %(raw)s
    """
    # Load/check file name
    raw = mne.io.read_raw_fif(fname, preload=True)

    # Rename channels
    try:
        mne.rename_channels(raw.info, {"AUX7": "EOG", "AUX8": "ECG"})
        logger.debug("Channels AUX7 and AUX8 found and renamed.")
    except Exception:
        mne.rename_channels(raw.info, {"AUX19": "EOG", "AUX20": "ECG"})
        logger.debug("Channels AUX19 and AUX20 found and renamed.")
    raw.set_channel_types(mapping={"ECG": "ecg", "EOG": "eog"})

    # Old eego LSL plugin has upper case channel names
    mapping = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "POZ": "POz",
        "FCZ": "FCz",
        "OZ": "Oz",
        "FPz": "Fpz",
    }
    for key, value in mapping.items():
        try:
            mne.rename_channels(raw.info, {key: value})
            logger.debug("Channel '%s' renamed to '%s'.", key, value)
        except Exception:
            pass

    return raw
