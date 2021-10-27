"""Processing of Tinnitus Handicap Inventory (THI) Evamed questionnaires."""

import pandas as pd


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

    # extract information
    valid_answers = {'No': 0, 'Sometimes': 2, 'Yes': 4}
    df_thi = df[[f'{prefix}_THI{k}' for k in range(1, 26)]]\
        .replace(valid_answers)
    df_thi.rename(mapper={f'{prefix}_THI{k}': f'Q{k}' for k in range(1, 26)},
                  axis='columns', copy=False, inplace=True)
    df_thi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_thi.insert(1, 'results', df[f'{prefix}_THI_R'])

    # sanity-check
    assert (df_thi[[f'Q{k}' for k in range(1, 26)]].sum(axis=1) == \
            df_thi['results']).all()

    return df_thi
