"""Parser of Tinnitus Handicap Inventory (THI) Evamed questionnaires."""

import re

import pandas as pd

from ...utils.checks import _check_participant, _check_participants


def parse_thi(df, participant):
    """Parse dataframe and extract THI answers and information.
    Assumes only one THI questionnaire is present in the dataframe,
    e.g. baseline."""
    _check_participant(participant)

    # clean-up columns
    columns = [col for col in df.columns if 'THI' in col]
    assert len(columns) != 0, 'THI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract questions
    pattern = re.compile(f'{prefix}_THI' + r'\d{1,2}')
    columns_questions = [col for col in columns if pattern.match(col)]
    valid_answers = {
        'No': 0,
        'Sometimes': 2,
        'Yes': 4
    }
    df_thi = df[columns_questions].replace(valid_answers)
    # extract date/results
    df_thi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_thi.insert(1, 'results', df[f'{prefix}_THI_R'])

    # sanity-check
    assert (df_thi[columns_questions].sum(axis=1) == df_thi['results']).all()

    # rename
    mapper = {col: 'Q'+re.search(r'\d{1,2}', col).group()
              for col in columns_questions}
    df_thi.rename(mapper=mapper, axis='columns', copy=False, inplace=True)

    # re-index
    df_thi.reset_index(drop=True, inplace=True)

    return df_thi


def parse_multi_thi(df, participants):
    """Parse the THI results from multiple THI questionnaires/participants."""
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
