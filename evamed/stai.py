"""Processing of State and Trait Anxiety Inventory (STAI) Evamed
questionnaires."""

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

    # extract information
    valid_answers = {
        '1. ALMOST NEVER': 1,
        '1. NOT AT ALL': 1,
        '2. SOMETIMES': 2,
        '2. SOMEWHAT': 2,
        '3. OFTEN': 3,
        '3. MODERATELY SO': 3,
        '4. VERY MUCH SO': 4,
        '4. ALMOST ALWAYS': 4
    }
    df_stai = df[[f'{prefix}_STAI{k}' for k in range(1, 41)]]\
        .replace(valid_answers)
    df_stai.rename(mapper={f'{prefix}_STAI{k}': f'Q{k}' for k in range(1, 41)},
                   axis='columns', copy=False, inplace=True)
    df_stai.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_stai.insert(1, 'results', df[f'{prefix}_STAI_R'])

    # sanity-check
    weights_4_to_1 = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30,
                      33, 34, 36, 39]
    weights_1_to_4 = [3, 4, 6, 7, 9, 12, 13, 14, 17, 18, 22, 24, 25, 28, 29,
                      31, 32, 35, 37, 38, 40]
    tmp = df_stai[[f'Q{k}' for k in weights_4_to_1]]\
        .replace({1: 4, 2: 3, 3: 2, 4: 1})
    tmp[[f'Q{k}' for k in weights_1_to_4]] = \
        df_stai[[f'Q{k}' for k in weights_1_to_4]]
    assert (tmp[[f'Q{k}' for k in range(1, 41)]].sum(axis=1) == \
            df_stai['results']).all()

    return df_stai
