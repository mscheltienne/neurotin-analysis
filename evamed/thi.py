"""Processing of Tinnitus Handicap Inventory (THI) Evamed questionnaires."""

import re

import pandas as pd

from utils import read_paths, read_csv


def _parse_thi(df, participant):
    """Parse dataframe and extract THI answers and information."""
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


def _parse_thi_evolution(participant):
    """
    Parse evolution of THI scores.
    """
    paths = read_paths()
    index = ('baseline', 'pre-assessment', 'post-assessment')
    dfs = {type_: read_csv(paths[type_]) for type_ in index}
    dfs_thi = {type_: _parse_thi(df, participant) for type_, df in dfs.items()}

    results = [dfs_thi[type_]['results'][0] for type_ in index]
    series = pd.Series(results, index)
    return series
