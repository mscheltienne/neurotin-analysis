"""Processing of Pittsburg Sleep Quality Index (PSQI) Evamed questionnaires."""

import re

import pandas as pd


def _parse_psqi(df, participant):
    """Parse dataframe and extract PSQI answers and information."""
    # clean-up columns
    columns = [col for col in df.columns if 'PSQI' in col]
    assert len(columns) != 0, 'PSQI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract information
    valid_answers = {
        'Not during the past month': 0,
        'No problem at all': 0,
        'Very good': 0,
        'No bed partner or room mate': 0,
        'Less than once a week': 1,
        'Only a very slight problem': 1,
        'Fairly good': 1,
        'Partner/room mate in other room': 1,
        'Once or twice a week': 2,
        'Fairly bad': 2,
        'Somewhat of a problem': 2,
        'Partner in same room, but not same bed': 2,
        'Three or more times a week': 3,
        'A very big problem': 3,
        'Very bad': 3,
        'Partner in same bed': 3
    }
    pattern = re.compile(rf'{prefix}_PSQI\d')
    col_questions = [col for col in columns if pattern.match(col)]
    df_psqi = df[col_questions].replace(valid_answers)
    df_psqi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_psqi.insert(1, 'results', df[f'{prefix}_PSQI_R'])

    return df_psqi
