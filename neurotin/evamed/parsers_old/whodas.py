"""Parser of WHO Disability Assessment Scale (WHODAS) Evamed questionnaires."""

import re

import pandas as pd

from ...utils.checks import _check_participant


def parse_whodas(df, participant):
    """Parse dataframe and extract WHODAS answers and information.
    Assumes only one WHODAS questionnaire is present in the dataframe,
    e.g. baseline."""
    _check_participant(participant)

    # clean-up columns
    columns = [col for col in df.columns if 'WHODAS' in col]
    assert len(columns) != 0, 'WHODAS not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract questions
    # d-section
    pattern = re.compile(rf'{prefix}_WHOD\d{2}')
    col_questions_section_d = [col for col in columns if pattern.match(col)]
    valid_answers = {
        'None': 1,
        'Mild': 2,
        'Moderate': 3,
        'Severe': 4,
        'Extreme or cannot do': 5  # to confirm
    }
    # h-section
    pattern = re.compile(rf'{prefix}_WHODH\d')
    col_questions_section_h = [col for col in columns if pattern.match(col)]

    df_whodas = pd.concat((df[col_questions_section_d].replace(valid_answers),
                           df[col_questions_section_h]),
                          axis=1)

    # extract date/results
    df_whodas.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_whodas.insert(1, 'results', df[f'{prefix}_WHODAS_R'])

    # rename d-section
    df_whodas.rename(mapper={col: f'Q{col[-2:]}D'
                             for col in col_questions_section_d},
                     axis='columns', copy=False, inplace=True)
    # rename h-section
    df_whodas.rename(mapper={col: f'Q{col[-1]}H'
                             for col in col_questions_section_h},
                     axis='columns', copy=False, inplace=True)

    # re-index
    df_whodas.reset_index(drop=True, inplace=True)

    return df_whodas
