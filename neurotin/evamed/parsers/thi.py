"""Parser of Tinnitus Handicap Inventory (THI) Evamed questionnaires.

Export:
    -> Analysis
    -> Select group
    -> Export
        -> CSV
        -> Synthesis
        -> [x] Export labels of choce
    -> Select all 3 questionnaires
        -> [x] Tinnitus Handicap Inventory (THI)
"""

import pandas as pd

from ...utils.checks import _check_participants


def parse_thi(df, participants):
    """Parse the THI from multiple THI questionnaires/participants
    that have different prefix, e.g. 'THI', 'THIB', 'THIPREA'."""
    _check_participants(participants)

    # clean-up columns
    columns = [col for col in df.columns if 'THI' in col]
    assert len(columns) != 0, 'THI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    df_thi_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            df_thi_dict['participant'].append(idx)
            df_thi_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            df_thi_dict['date'].append(date)
            reslt = df.loc[df['patient_code'] == idx, f'{pre}_THI_R'].values[0]
            df_thi_dict['result'].append(reslt)

    df_thi = pd.DataFrame.from_dict(df_thi_dict)
    df_thi.date = pd.to_datetime(df_thi.date)

    # rename
    mapper = {'THIB': 'Baseline',
              'THIPREA': 'Pre-assessment',
              'THI': 'Post-assessment'}
    df_thi['prefix'].replace(to_replace=mapper, inplace=True)
    df_thi.rename(columns=dict(prefix='When'), inplace=True)

    return df_thi
