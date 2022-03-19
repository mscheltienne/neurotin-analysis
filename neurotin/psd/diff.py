from itertools import product

import numpy as np
import pandas as pd

from ..utils._checks import _check_type
from ..utils._docs import fill_doc


@fill_doc
def diff_between_phases(df, column='avg'):
    """
    Compute the difference between a column in a regulation phase and in the
    preceding non-regulation phase.

    Parameters
    ----------
    %(df_psd)s
    column : str
        Label of the column on which the difference is computed between both
        phases.

    Returns
    -------
    df : DataFrame
        Difference between the average band power in the regulation phase and
        the preceding non-regulation phase.
            participant : int - Participant ID
            session : int - Session ID (1 to 15)
            run : int - Run ID
            idx : ID of the phase within the run (0 to 9)
            diff : float - PSD difference
    """
    _check_type(column, (str, ), item_name='column')
    assert column in df.columns

    # check keys
    keys = ['participant', 'session', 'run', 'idx']
    assert len(set(keys).intersection(df.columns)) == len(keys)

    # container for new df with diff between phases
    data = {key: [] for key in keys + ['diff']}

    participants = sorted(df['participant'].unique())
    for participant in participants:
        df_participant = df[df['participant'] == participant]

        sessions = sorted(df_participant['session'].unique())
        for session in sessions:
            df_session = df_participant[df_participant['session'] == session]

            runs = sorted(df_session['run'].unique())
            for run in runs:
                df_run = df_session[df_session['run'] == run]

                index = sorted(df_session['idx'].unique())
                for idx in index:
                    df_idx = df_run[df_run['idx'] == idx]

                    # compute the difference between regulation and rest
                    reg = df_idx[df_idx.phase == 'regulation'][column]
                    non_reg = df_idx[df_idx.phase == 'non-regulation'][column]
                    try:
                        diff = reg.values[0] - non_reg.values[0]
                    except IndexError:
                        continue

                    # fill dict
                    data['participant'].append(participant)
                    data['session'].append(session)
                    data['run'].append(run)
                    data['idx'].append(idx)
                    data['diff'].append(diff)

    # create df
    df = pd.DataFrame.from_dict(data, orient='columns')
    return df


def count_diff(df):
    """
    Count the positive/negative diff values by session.
    The count is normalized by the number of observations.

    Parameters
    ----------
    df : DataFrame
        Difference between the average band power in the regulation phase and
        the preceding non-regulation phase.
            participant : int - Participant ID
            session : int - Session ID (1 to 15)
            run : int - Run ID
            idx : ID of the phase within the run (0 to 9)
            diff : float - PSD difference

    Returns
    -------
    df_positives : DataFrame
        Counts of positives 'diff'.
    df_negatives : DataFrame
        Counts of negatives 'diff'.
    """
    assert 'diff' in df.columns

    # reset order
    df = df.sort_values(by=['participant', 'session', 'run', 'idx'],
                        ascending=True)
    df.reset_index()

    # check sign
    df['sign'] = np.sign(df['diff'])
    # groupby
    counts = df.groupby(by=['participant', 'session'],
                        dropna=True)['sign'].value_counts()

    # create new counts dataframe
    positives = {'participant': [], 'session': [], 'count': []}
    negatives = {'participant': [], 'session': [], 'count': []}

    participants = df['participant'].unique()
    sessions = df['session'].unique()
    for participant, session in product(participants, sessions):
        try:
            pos = counts[participant, session, 1]
            neg = counts[participant, session, -1]
        except KeyError:
            pos = np.nan
            neg = np.nan

        # add common data to dict
        positives['participant'].append(participant)
        positives['session'].append(session)
        negatives['participant'].append(participant)
        negatives['session'].append(session)

        if any(np.isnan(x) or x == 0 for x in (pos, neg)):
            positives['count'].append(np.nan)
            negatives['count'].append(np.nan)
        else:
            # normalize
            total = (pos + neg)
            pos = pos / total
            neg = neg / total
            # add to dict
            positives['count'].append(pos)
            negatives['count'].append(-neg)

    # create df
    df_positives = pd.DataFrame.from_dict(positives, orient='columns')
    df_negatives = pd.DataFrame.from_dict(negatives, orient='columns')

    # reset order
    df_positives.sort_values(by=['participant', 'session'], ascending=True,
                             inplace=True)
    df_positives.reset_index()
    df_negatives.sort_values(by=['participant', 'session'], ascending=True,
                             inplace=True)
    df_negatives.reset_index()

    return df_positives, df_negatives
