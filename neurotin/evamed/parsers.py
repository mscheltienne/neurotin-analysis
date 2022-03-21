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


@fill_doc
def parse_bdi(df, participants):
    """Parse the BDI date/results from multiple BDI questionnaires and
    participants. The input .csv file can be obtained by exporting from evanmed
    with the following settings:
        -> Analysis
        -> Select group
        -> Export
            -> CSV
            -> Synthesis
            -> [x] Export labels of choce
        -> Select all 2 questionnaires
            -> [x] Beck's Depression Inventory (BDI)

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
    columns = [col for col in df.columns if 'BDI' in col]
    assert len(columns) != 0, 'BDI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    bdi_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            bdi_dict['participant'].append(idx)
            bdi_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            bdi_dict['date'].append(date)
            result = df.loc[df['patient_code'] == idx,
                            f'{pre}_BDI_R'].values[0]
            bdi_dict['result'].append(result)

    bdi = pd.DataFrame.from_dict(bdi_dict)
    bdi.date = pd.to_datetime(bdi.date)

    # rename
    mapper = {'BDIB': 'Baseline',
              'BDIV': 'Post-assessment'}
    bdi['prefix'].replace(to_replace=mapper, inplace=True)
    bdi.rename(columns=dict(prefix='visit'), inplace=True)

    return bdi


@fill_doc
def parse_psqi(df, participants):
    """Parse the PSQI date/results from multiple PSQI questionnaires and
    participants. The input .csv file can be obtained by exporting from evanmed
    with the following settings:
        -> Analysis
        -> Select group
        -> Export
            -> CSV
            -> Synthesis
            -> [x] Export labels of choce
        -> Select all 2 questionnaires
            -> [x] Pittsburgh Sleep Quality Index (PSQI)

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
    columns = [col for col in df.columns if 'PSQI' in col]
    assert len(columns) != 0, 'PSQI not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    psqi_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            psqi_dict['participant'].append(idx)
            psqi_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            psqi_dict['date'].append(date)
            result = df.loc[df['patient_code'] == idx,
                            f'{pre}_PSQI_R'].values[0]
            psqi_dict['result'].append(result)

    psqi = pd.DataFrame.from_dict(psqi_dict)
    psqi.date = pd.to_datetime(psqi.date)

    # rename
    mapper = {'PSQIB': 'Baseline',
              'PSQI': 'Post-assessment'}
    psqi['prefix'].replace(to_replace=mapper, inplace=True)
    psqi.rename(columns=dict(prefix='visit'), inplace=True)

    return psqi


@fill_doc
def parse_whodas(df, participants):
    """Parse the WHODAS date/results from multiple WHODAS questionnaires and
    participants. The input .csv file can be obtained by exporting from evanmed
    with the following settings:
        -> Analysis
        -> Select group
        -> Export
            -> CSV
            -> Synthesis
            -> [x] Export labels of choce
        -> Select all 2 questionnaires
            -> [x] WHO Disability Assessment Scale (WHODAS)

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
    columns = [col for col in df.columns if 'WHODAS' in col]
    assert len(columns) != 0, 'WHODAS not present in dataframe.'
    prefix = set(col.split('_')[0] for col in columns)
    assert len(prefix) != 0  # sanity-check

    whodas_dict = dict(participant=[], prefix=[], date=[], result=[])
    for idx in participants:
        for pre in prefix:
            whodas_dict['participant'].append(idx)
            whodas_dict['prefix'].append(pre)
            date = df.loc[df['patient_code'] == idx, f'{pre}_date'].values[0]
            whodas_dict['date'].append(date)
            result = df.loc[df['patient_code'] == idx,
                            f'{pre}_WHODAS_R'].values[0]
            whodas_dict['result'].append(result)

    whodas = pd.DataFrame.from_dict(whodas_dict)
    whodas.date = pd.to_datetime(whodas.date)

    # rename
    mapper = {'WHODASB': 'Baseline',
              'WHODAS': 'Post-assessment'}
    whodas['prefix'].replace(to_replace=mapper, inplace=True)
    whodas.rename(columns=dict(prefix='visit'), inplace=True)

    return whodas
