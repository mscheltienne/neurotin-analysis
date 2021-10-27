"""Processing of WHO Disability Assessment Scale (WHODAS) Evamed
questionnaires."""

import re

import pandas as pd


def _parse_whodas(df, participant):
    """Parse dataframe and extract WHODAS answers and information."""
    # clean-up columns
    columns = [col for col in df.columns if 'WHODAS' in col]
    assert len(columns) != 0, 'WHODAS not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract information
    # d-section
    pattern = re.compile(rf'{prefix}_WHOD\d\d')
    col_d = [col for col in columns if pattern.match(col)]
    valid_answers = {'None': 1,
                     'Mild': 2,
                     'Moderate': 3,
                     'Severe': 4,
                     'Extreme or cannot do': 5}  # to confirm
    # h-section
    pattern = re.compile(rf'{prefix}_WHODH\d')
    col_h = [col for col in columns if pattern.match(col)]

    df_whodas = pd.concat((df[col_d].replace(valid_answers), df[col_h]),
                          axis=1)

    # rename d-section
    df_whodas.rename(mapper={col: f'D-Q{col[-2:]}' for col in col_d},
                     axis='columns', copy=False, inplace=True)
    # rename h-section
    df_whodas.rename(mapper={col: f'H-Q{col[-1]}' for col in col_h},
                     axis='columns', copy=False, inplace=True)

    df_whodas.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_whodas.insert(1, 'results', df[f'{prefix}_WHODAS_R'])

    return df_whodas
