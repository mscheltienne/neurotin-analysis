from pathlib import Path

import mne


def list_raw_fif(directory, exclude=[]):
    """
    List all raw fif files in directory and its subdirectories.

    Parameters
    ----------
    directory : str | Path
        Path to the directory.
    exclude : list | tuple
        List of files to exclude.

    Returns
    -------
    fifs : list
        Found raw fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_raw_fif(elt))
        elif elt.name.endswith("-raw.fif") and elt not in exclude:
            fifs.append(elt)
    return fifs


def read_exclusion(exclusion_file):
    """
    Read the list of input fif files to exclude from preprocessing.
    If the file storing the exlusion list does not exist, it is created.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the path to input files to exclude.

    Returns
    -------
    exclude : list
        List of files to exclude.
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
    Add a fif file or a set of fif files to the exclusion file.

    Parameters
    ----------
    exclusion_file : str | Path
        Text file storing the path to input files to exclude.
    exclude : str | Path | list | tuple
        Path or list of Paths to input files to exclude.
    """
    exclusion_file = Path(exclusion_file)
    mode = 'w' if not exclusion_file.exists() else 'a'
    if isinstance(exclude, (str, Path)):
        exclude = [str(exclude)] if Path(exclude).exists() else []
    elif isinstance(exclude, (list, tuple)):
        exclude = [str(fif) for fif in exclude if Path(fif).exists()]
    with open(exclusion_file, mode) as file:
        for fif in exclude:
            file.write(str(fif) + '\n')


def parse_subject_info(fname):
    """
    Parse the subject_info file and return the subject ID and sex.

    Parameters
    ----------
    fname : str | Path
        Path to the subject info file.

    Returns
    -------
    dict
        key : int
            ID of the subject.
        value : tuple (sex, )
            sex : int
                Sex of the subject. 1: Male - 2: Female.
    """
    fname = Path(fname)
    assert fname.exists()
    with open(fname, 'r') as file:
        lines = file.readlines()
    lines = [line.strip().split(';') for line in lines if len(line) > 0]
    lines = [[eval(l.strip()) for l in line]
             for line in lines if len(line) == 3]
    return {line[0]: (line[1], line[2]) for line in lines}


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
    raw : Raw
        Raw instance.
    """
    # Load/check file name
    raw = mne.io.read_raw_fif(fname, preload=True)

    # Rename channels
    try:
        mne.rename_channels(raw.info, {"AUX7": "EOG", "AUX8": "ECG"})
    except Exception:
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
        except Exception:
            pass

    return raw
