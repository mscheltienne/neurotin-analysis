"""Processing of Beck's Depression Inventory (BDI) Evamed questionnaires."""

import pandas as pd


def _parse_thi(df, participant):
    """Parse dataframe and extract BDI answers and information."""
    # clean-up columns
    columns = [col for col in df.columns if 'BDI' in col]
    assert len(columns) != 0, 'BDI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) == 1
    prefix = list(prefix)[0]

    # locate participant lines
    df = df.loc[df['patient_code'] == participant]

    # extract information
    df_bdi = df[[f'{prefix}_BDI{k}' for k in range(1, 22)]]\
        .applymap(lambda x: int(x[0]))
    df_bdi.rename(mapper={f'{prefix}_BDI{k}': f'Q{k}' for k in range(1, 22)},
                  axis='columns', copy=False, inplace=True)
    df_bdi.insert(0, 'date', pd.to_datetime(df[f'{prefix}_date']))
    df_bdi.insert(1, 'results', df[f'{prefix}_BDI_R'])
    df_bdi.insert(2, 'marital status', df[f'{prefix}_MARISTA'])
    df_bdi.insert(3, 'occupation', df[f'{prefix}_OCCUP'])
    df_bdi.insert(4, 'education', df[f'{prefix}_EDUC'])

    # sanity-check
    assert (df_bdi[[f'Q{k}' for k in range(1, 22)]].sum(axis=1) == \
            df_bdi['results']).all()

    return df_bdi
