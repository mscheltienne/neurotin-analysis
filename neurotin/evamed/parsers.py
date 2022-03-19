import pandas as pd

from ..utils._checks import _check_participants
from ..utils._docs import fill_doc


@fill_doc
def parse_thi(df, participants):
    """Parse the THI date/results from multiple THI questionnaires and
    participants. The input .csv file can be obtained by exporting from evanmed
    with the following settings:
        -> Analysis
        -> Select group
        -> Export
            -> CSV
            -> Synthesis
            -> [x] Export labels of choce
        -> Select all 3 questionnaires
            -> [x] Tinnitus Handicap Inventory (THI)

    Parameters
    ----------
    %(df_raw_evamed)s
    %(participants)s

    Returns
    -------
    %(df_clinical)s
    """
    _check_participants(participants)

    # clean-up columns
    columns = [col for col in df.columns if 'THI' in col]
    assert len(columns) != 0, 'THI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    thi_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            thi_dict['participant'].append(idx)
            thi_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            thi_dict['date'].append(date)
            result = df.loc[df['patient_code'] == idx,
                            f'{pre}_THI_R'].values[0]
            thi_dict['result'].append(result)

    thi = pd.DataFrame.from_dict(thi_dict)
    thi.date = pd.to_datetime(thi.date)

    # rename
    mapper = {'THIB': 'Baseline',
              'THIPREA': 'Pre-assessment',
              'THI': 'Post-assessment'}
    thi['prefix'].replace(to_replace=mapper, inplace=True)
    thi.rename(columns=dict(prefix='visit'), inplace=True)

    return thi


@fill_doc
def parse_stai(df, participants):
    """Parse the STAI date/results from multiple STAI questionnaires and
    participants. The input .csv file can be obtained by exporting from evanmed
    with the following settings:
        -> Analysis
        -> Select group
        -> Export
            -> CSV
            -> Synthesis
            -> [x] Export labels of choce
        -> Select all 2 questionnaires
            -> [x] State and Trait Anxiety Inventory (STAI)

    Parameters
    ----------
    %(df_raw_evamed)s
    %(participants)s

    Returns
    -------
    %(df_clinical)s
    """
    _check_participants(participants)

    # clean-up columns
    columns = [col for col in df.columns if 'STAI' in col]
    assert len(columns) != 0, 'STAI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    stai_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            stai_dict['participant'].append(idx)
            stai_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            stai_dict['date'].append(date)
            result = df.loc[df['patient_code'] == idx,
                            f'{pre}_STAI_R'].values[0]
            stai_dict['result'].append(result)

    stai = pd.DataFrame.from_dict(stai_dict)
    stai.date = pd.to_datetime(stai.date)

    # rename
    mapper = {'STAIB': 'Baseline',
              'STAI': 'Post-assessment'}
    stai['prefix'].replace(to_replace=mapper, inplace=True)
    stai.rename(columns=dict(prefix='visit'), inplace=True)

    return stai
