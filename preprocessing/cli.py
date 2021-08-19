import os
from pathlib import Path

from pipeline import preprocessing_pipeline


FOLDER_IN = Path(r'/Volumes/NeuroTin-EEG/Data/Participants')
FOLDER_OUT = Path(r'/Volumes/NeuroTin-EEG/Data preprocessed/')
RETRIES=3


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
        if participant <= 0:
            raise ValueError
        if str(participant).zfill(3) not in os.listdir(FOLDER_IN):
            raise ValueError

    for _ in range(RETRIES):
        try:
            participant = _check_participant(input('[IN] Participant ID: '))
            break
        except ValueError:
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
        List of found raw fif files.
    """
    directory = Path(directory)
    fifs = list()
    for elt in directory.iterdir():
        if elt.is_dir():
            fifs.extend(list_raw_fif(directory/elt))
        elif elt.name.endswith('-raw.fif'):
            fifs.append(elt)
    return fifs


def main():
    """
    Main preprocessing pipeline, called once per participant.
    """
    _, participant_folder = input_participant()
    dirname_in = FOLDER_OUT / participant_folder
    dirname_out = FOLDER_OUT / participant_folder
    if not dirname_in.exists():
        raise ValueError
    if not dirname_out.exists():
        os.makedirs(dirname_out)

    fifs = list_raw_fif(dirname_in)
    for fif_in in fifs:
        fif_out = dirname_out / fif_in.relative_to(dirname_in)
        if fif_out.exists():
            continue
        print ('-------------------------------------------------------------')
        print (f'Preprocesing {fif_in.relative_to(dirname_in)}')
        raw = preprocessing_pipeline(fif_in)
        raw.save(fif_out, fmt='double')


if __name__ == '__main__':
    main()
