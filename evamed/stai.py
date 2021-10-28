"""Processing of State and Trait Anxiety Inventory (STAI) Evamed
questionnaires."""

import re

import pandas as pd


def _parse_stai(df, participant):
    """Parse dataframe and extract STAI answers and information."""
    # clean-up columns
    columns = [col for col in df.columns if 'STAI' in col]
    assert len(columns) != 0, 'STAI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract questions
    pattern = re.compile(f'{prefix}_STAI' + r'\d{1,2}')
    columns_questions = [col for col in columns if pattern.match(col)]
    df_stai = df[columns_questions].applymap(lambda x: int(x[0]),
                                             na_action='ignore')
    # extract date/results
    df_stai.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_stai.insert(1, 'results', df[f'{prefix}_STAI_R'])

    # sanity-check
    questions_weights_4_to_1 = [
        f'{prefix}_STAI'+str(k)
        for k in (1, 2, 5, 8, 10, 11, 15, 16, 19, 20,
                  21, 23, 26, 27, 30, 33, 34, 36, 39)
    ]
    questions_weights_1_to_4 = [
        f'{prefix}_STAI'+str(k)
        for k in (3, 4, 6, 7, 9, 12, 13, 14, 17, 18, 22,
                  24, 25, 28, 29,31, 32, 35, 37, 38, 40)
    ]
    tmp = df_stai[questions_weights_4_to_1].replace({1: 4, 2: 3, 3: 2, 4: 1})
    tmp[questions_weights_1_to_4] = df_stai[questions_weights_1_to_4]
    assert (tmp[columns_questions].sum(axis=1) == df_stai['results']).all()

    # rename
    mapper = {col: 'Q'+re.search(r'\d{1,2}', col).group()
              for col in columns_questions}
    df_stai.rename(mapper=mapper, axis='columns', copy=False, inplace=True)

    # re-index
    df_stai.reset_index(drop=True, inplace=True)

    return df_stai
