"""Parser of Beck's Depression Inventory (BDI) Evamed questionnaires."""

import re

import pandas as pd

from ...utils.checks import _check_participant


def parse_bdi(df, participant):
    """Parse dataframe and extract BDI answers and information."""
    _check_participant(participant)

    # clean-up columns
    columns = [col for col in df.columns if 'BDI' in col]
    assert len(columns) != 0, 'BDI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract questions
    pattern = re.compile(f'{prefix}_BDI' + r'\d{1,2}')
    columns_questions = [col for col in columns if pattern.match(col)]
    df_bdi = df[columns_questions].applymap(lambda x: int(x[0]),
                                            na_action='ignore')
    # extract date/results
    df_bdi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_bdi.insert(1, 'results', df[f'{prefix}_BDI_R'])
    # extract additional information
    df_bdi.insert(2, 'marital status', df[f'{prefix}_MARISTA'])
    df_bdi.insert(3, 'occupation', df[f'{prefix}_OCCUP'])
    df_bdi.insert(4, 'education', df[f'{prefix}_EDUC'])

    # sanity-check
    assert (df_bdi[columns_questions].sum(axis=1) == df_bdi['results']).all()

    # rename
    mapper = {col: 'Q'+re.search(r'\d{1,2}', col).group()
              for col in columns_questions}
    df_bdi.rename(mapper=mapper, axis='columns', copy=False, inplace=True)

    # re-index
    df_bdi.reset_index(drop=True, inplace=True)

    return df_bdi
