import pickle
import datetime

from ..utils.checks import _check_type, _check_path


def load_model(folder, participant, session, model_idx='auto'):
    """
    Load a saved model for a given participant/session.
    """
    folder = _check_path(folder, item_name='folder', must_exist=True)
    participant, participant_folder = _check_participant(participant)
    session = _check_session(session)
    model_idx = _check_model_idx(model_idx)

    session_dir = folder/participant_folder/f'Session {session}'

    if model_idx == 'auto':
        # read_logs
        logs = [log for log in _read_logs(session_dir) if log[1] == 'Model']
        valid_model_idx = [int(log[2].split(' ')[2])
                           for log in logs if len(log) == 3]
        assert 0 < len(valid_model_idx)
        model_idx = max(valid_model_idx)

    model_fname = session_dir/'Model'/f'{model_idx}-model.pcl'

    with open(model_fname, 'rb') as f:
        weights, info, reject, reject_local, calib_idx = pickle.load(f)

    return weights, info, reject, reject_local, calib_idx


def _check_participant(participant):
    """Check argument participant."""
    _check_type(participant, ('int', ), item_name='participant')
    assert 50 <= participant <= 150, 'Invalid participant ID.'
    return participant, str(participant).zfill(3)


def _check_session(session):
    """Check argument session."""
    _check_type(session, ('int', ), item_name='session')
    assert 1 <= session <= 15, 'Invalid session ID.'
    return session


def _check_model_idx(model_idx):
    """Check argument model_idx."""
    _check_type(model_idx, ('int', str), item_name='model_idx')
    if isinstance(model_idx, str):
        model_idx = model_idx.lower().strip()
        assert model_idx == 'auto', 'Invalid model IDx.'
    else:
        assert 1 <= model_idx, 'Invalid model IDx.'
    return model_idx


def _read_logs(session_dir):
    """Read logs for a given participant/session."""
    session_dir = _check_path(session_dir, item_name='session_dir',
                              must_exist=True)
    logs_file = _check_path(session_dir/'logs.txt', item_name='logs_file',
                            must_exist=True)

    with open(logs_file, 'r') as f:
        lines = f.readlines()

    lines = [line.split(' - ') for line in lines if len(line.split(' - ')) > 1]
    logs = [[datetime.strptime(line[0].strip(), "%d/%m/%Y %H:%M")]+
            [line[k].strip() for k in range(1, len(line))] for line in lines]

    return sorted(logs, key=lambda x:x[0], reverse=False)