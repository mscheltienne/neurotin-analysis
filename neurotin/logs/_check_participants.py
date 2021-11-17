from ..utils.checks import _check_type, _check_participant


def _check_participants(participants):
    """Check the participant(s) and return it as a list of participants."""
    _check_type(participants, ('int', list, tuple), item_name='participants')

    if isinstance(participants, (int, )):
        participants = [participants]
    elif isinstance(participants, (tuple, )):
        participants = list(participants)

    for participant in participants:
        _check_participant(participant)

    return participants
