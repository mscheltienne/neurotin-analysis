"""Parser of Pittsburg Sleep Quality Index (PSQI) Evamed questionnaires."""

import re
import copy

import pandas as pd

from ...utils.checks import _check_participant


def parse_psqi(df, participant):
    """Parse dataframe and extract PSQI answers and information."""
    _check_participant(participant)

    # clean-up columns
    columns = [col for col in df.columns if 'PSQI' in col]
    assert len(columns) != 0, 'PSQI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract questions
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
    columns_questions = [col for col in columns if pattern.match(col)]
    df_psqi = df[columns_questions].replace(valid_answers)
    # extract date/results
    df_psqi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_psqi.insert(1, 'results', df[f'{prefix}_PSQI_R'])

    # rename
    mapper = {col: 'Q'+col.split(f'{prefix}_PSQI')[1].lower()
              for col in columns_questions}
    df_psqi.rename(mapper=mapper, axis='columns', copy=False, inplace=True)
    df_psqi.rename(mapper={'Q5j2': 'Q5j-(opt)', 'Q10e2': 'Q10e-(opt)'},
                   axis='columns', copy=False, inplace=True)

    # reorder
    old_columns = df_psqi.columns.tolist()
    columns_to_check = [('Q5j', 'Q5j-(opt)'), ('Q10e', 'Q10e-(opt)')]
    new_columns = copy.deepcopy(columns)
    for first, second in columns_to_check:
        try:
            first_idx = old_columns.index(first)
            second_idx = old_columns.index(second)
            assert abs(first_idx-second_idx) == 1
            if second_idx < first_idx:
                new_columns[first_idx], new_columns[second_idx] = \
                    new_columns[second_idx], new_columns[first_idx]
        except ValueError:
            continue
    df_psqi = df_psqi.reindex(new_columns, axis=1)

    # re-index
    df_psqi.reset_index(drop=True, inplace=True)

    return df_psqi
