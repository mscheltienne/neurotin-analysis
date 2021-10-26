import pandas as pd


def _parse_thi(df, participant):
    """Parse dataframe and extract THI answers and information."""
    thi_columns = [col for col in df.columns if 'THI' in col]
    assert len(thi_columns) != 0, 'THI not present in dataframe.'
    df = df.loc[df['patient_code'] == participant]

    valid_answers = {'No': 0, 'Sometimes': 2, 'Yes': 4}
    df_thi = df[[f'THIB_THI{k}' for k in range(1, 26)]]\
        .replace(valid_answers)
    df_thi.rename(mapper={f'THIB_THI{k}': f'Q{k}' for k in range(1, 26)},
                  axis='columns', copy=False, inplace=True)
    df_thi.insert(0, 'date', pd.to_datetime(df.THIB_date))
    df_thi.insert(1, 'results', df.THIB_THI_R)

    assert (df_thi[[f'Q{k}' for k in range(1, 26)]].sum(axis=1) == \
            df_thi['results']).all()

    return df_thi
