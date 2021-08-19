import os
from pathlib import Path


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


def main():
    _, participant_folder = input_participant()
    dirname = FOLDER_OUT / participant_folder
    if not dirname.exists():
        os.makedirs(dirname)




if __name__ == '__main__':
    pass
