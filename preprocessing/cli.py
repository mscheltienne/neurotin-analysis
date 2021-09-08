import os
import sys


RETRIES = 3


def input_participant(folder_in):
    """
    Input a participant ID.

    Parameters
    ----------
    folder_in : str | Path
        Path to the folder containing the participant folders.

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
        assert str(participant).zfill(3) in os.listdir(folder_in)
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


def query_yes_no(question, default="yes", retries=RETRIES):
    """
    Ask a yes/no question via input() and return their answer.

    Parameters
    ----------
    question : str
        String that is presented to the user.
    default : str
        Presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    retries : int
        Number of invalid input that can be given until an error is raised.

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    attempt = 1
    while attempt <= RETRIES:
        choice = input("[IN] " + question + prompt).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write(
                "Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
        attempt += 1
    else:
        raise ValueError
