from pathlib import Path

import mne


def list_raw_fif(directory):
    """
    List all raw fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory.

    Returns
    -------
    fifs : list
        Found raw fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_raw_fif(directory / elt))
        elif elt.name.endswith("-raw.fif"):
            fifs.append(elt)
    return fifs


def read_exclusion(exclusion_file):
    """
    Read the list of output fif files to exclude from disk.
    If the file storing the exlusion list does not exist, it is created at
    'FOLDER_OUT / exlusion.txt'.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the file output path to exclude.

    Returns
    -------
    exclude : list
        List of file to exclude.
    """
    exclusion_file = Path(exclusion_file)
    if exclusion_file.exists():
        with open(exclusion_file, 'r') as file:
            exclude = file.readlines()
        exclude = [line.rstrip() for line in exclude if len(line) > 0]
    else:
        with open(exclusion_file, 'w'):
            pass
        exclude = list()
    return [Path(file) for file in exclude]


def write_exclusion(exclusion_file, exclude):
    """
    Add a file to the exclusion file 'FOLDER_OUT / exlusion.txt'.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the file output path to exclude.
    exclude : str | Path | list | tuple
        Path or list of Paths to the output file to exclude.
    """
    exclusion_file = Path(exclusion_file)
    mode = 'w' if not exclusion_file.exists() else 'a'
    if isinstance(exclude, (str, Path)):
        exclude = [Path(exclude)]
    elif isinstance(exclude, (list, tuple)):
        exclude = [Path(fif) for fif in exclude]
    with open(exclusion_file, mode) as file:
        for fif in exclude:
            file.write(str(fif) + '\n')


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

    # Subject info
    participant = int(fname.parent.parent.parent.name)
    session = int(fname.parent.parent.name.split()[-1])
    recording_type = fname.parent.name
    recording_run = fname.name.split('-')[0]
    description = f'Session {session} - {recording_type} {recording_run}'
    raw.info['subject_info'] = dict()
    raw.info['subject_info']['id'] = participant
    raw.info['subject_info']['his_id'] = str(participant).zfill(3)
    raw.info['description'] = description

    # Device info
    raw.info['device_info'] = dict()
    raw.info['device_info']['type'] = 'EEG'
    raw.info['device_info']['model'] = 'eego mylab'
    serial = fname.stem.split('-raw')[0].split('-')[-1].split()[1]
    raw.info['device_info']['serial'] = serial
    raw.info['device_info']['site'] = \
        'https://www.ant-neuro.com/products/eego_mylab'

    return raw
