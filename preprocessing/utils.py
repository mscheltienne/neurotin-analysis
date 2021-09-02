import mne


def read_raw_fif(fname):
    """
    Load a RAW instance from a .fif file. Renames the channel to match the
    standard 10/20 convention. Rename the AUX channels to ECG and EOG. Add the
    reference channel 'CPz' and add the standard 1020 Dig montage.

    Parameters
    ----------
    fname : str | Path
        Path to the MNE raw file to read. Must be in .fif format.

    Returns
    -------
    raw : Raw instance.
    """
    # Load/check file name
    raw = mne.io.read_raw_fif(fname, preload=True)

    # Rename channels
    try:
        mne.rename_channels(raw.info, {"AUX7": "EOG", "AUX8": "ECG"})
    except:
        mne.rename_channels(raw.info, {"AUX19": "EOG", "AUX20": "ECG"})
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
        except:
            pass

    return raw
