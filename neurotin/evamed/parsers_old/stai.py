"""Parser of State and Trait Anxiety Inventory (STAI) Evamed questionnaires.

Export:
    -> Analysis
    -> Select group
    -> Export
        -> CSV
        -> Synthesis
        -> [x] Export labels of choce
    -> Select all 2 questionnaires
        -> [x] State and Trait Anxiety Inventory (STAI)
"""

import pandas as pd

from ...utils.checks import _check_participants


def parse_stai(df, participants):
    """Parse dataframe and extract STAI answers and information.
    Assumes only one STAI questionnaire is present in the dataframe,
    e.g. baseline."""

    """Parse the STAI from multiple STAI questionnaires/participants
    that have different prefix, e.g. 'STAI', 'STAIB'."""
    _check_participants(participants)

    # clean-up columns
    columns = [col for col in df.columns if 'STAI' in col]
    assert len(columns) != 0, 'STAI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    df_stai_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            df_stai_dict['participant'].append(idx)
            df_stai_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            df_stai_dict['date'].append(date)
            reslt = df.loc[df['patient_code'] == idx,
                           f'{pre}_STAI_R'].values[0]
            df_stai_dict['result'].append(reslt)

    # code to compute the result score
    """
    # sanity-check
    questions_weights_4_to_1 = [
        f'{prefix}_STAI'+str(k)
        for k in (1, 2, 5, 8, 10, 11, 15, 16, 19, 20,
                  21, 23, 26, 27, 30, 33, 34, 36, 39)
    ]
    questions_weights_1_to_4 = [
        f'{prefix}_STAI'+str(k)
        for k in (3, 4, 6, 7, 9, 12, 13, 14, 17, 18, 22,
                  24, 25, 28, 29, 31, 32, 35, 37, 38, 40)
    ]
    tmp = df_stai[questions_weights_4_to_1].replace({1: 4, 2: 3, 3: 2, 4: 1})
    tmp[questions_weights_1_to_4] = df_stai[questions_weights_1_to_4]
    assert (tmp[columns_questions].sum(axis=1) == df_stai['results']).all()
    """

    df_stai = pd.DataFrame.from_dict(df_stai_dict)
    df_stai.date = pd.to_datetime(df_stai.date)

    # rename
    mapper = {'THIB': 'Baseline',
              'THIPREA': 'Pre-assessment',
              'THI': 'Post-assessment'}
    df_stai['prefix'].replace(to_replace=mapper, inplace=True)
    df_stai.rename(columns=dict(prefix='When'), inplace=True)

    return df_stai
