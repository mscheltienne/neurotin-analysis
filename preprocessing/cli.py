import os
from pathlib import Path

from pipeline import preprocessing_pipeline


FOLDER_IN = Path(r"/Users/scheltie/Documents/NeuroTin Data/Raw/")
FOLDER_OUT = Path(r"/Users/scheltie/Documents/NeuroTin Data/Clean/")
RETRIES = 3


def main():
    """
    Main preprocessing pipeline, called once per participant.
    """
    _, participant_folder = input_participant()
    dirname_in = FOLDER_IN / participant_folder
    dirname_out = FOLDER_OUT / participant_folder
    assert dirname_in.exists()
    os.makedirs(dirname_out, exist_ok=True)
    exclude = read_exclusion()

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
        except AssertionError:
            exclude.append(fif_out)
            write_exclusion(fif_out)
        raw.save(fif_out, fmt="double")


def input_participant():
    """
    Input a participant ID.

    Returns
    -------
    participant : int
        ID of the participant.
    participant_folder : str
        Folder of the participant. The name is created from the participant ID
        with str(participant).zfill(3).
    """

    def _check_participant(participant):
        participant = int(participant)
        assert 0 < participant
        assert str(participant).zfill(3) in os.listdir(FOLDER_IN)
        return participant

    for _ in range(RETRIES):
        try:
            participant = _check_participant(input("[IN] Participant ID: "))
            break
        except AssertionError:
            pass
    else:
        raise ValueError

    return participant, str(participant).zfill(3)


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


def read_exclusion():
    """
    Read the list of output fif files to exclude from disk.
    If the file storing the exlusion list does not exist, it is created at
    'FOLDER_OUT / exlusion.txt'.

    Returns
    -------
    exclude : list
        List of file to exclude.
    """
    fname = FOLDER_OUT / 'exlusion.txt'
    if fname.exists():
        with open(fname, 'r') as file:
            exclude = file.readlines()
        exclude = [line.rstrip() for line in exclude if len(line) > 0]
    else:
        with open(fname, 'w'):
            pass
        exclude = list()
    return exclude


def write_exclusion(fif_out):
    """
    Add a file to the exclusion file 'FOLDER_OUT / exlusion.txt'.

    Parameters
    ----------
    fif_out : str | Path
        Path to the output file to exclude.
    """
    fname = FOLDER_OUT / 'exlusion.txt'
    if not fname.exists():
        mode = 'w'
    else:
        mode = 'a'
    with open(fname, mode) as file:
        file.write(str(fif_out) + '\n')


if __name__ == "__main__":
    main()
