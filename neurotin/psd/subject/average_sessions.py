import numpy as np
import pandas as pd


from ...utils.docs import fill_doc


@fill_doc
def count_diff(df):
    """
    Count the positive/negative diff values for all sessions.
    The count is normalized by the number of observations.

    Parameters
    ----------
    %(psd_diff_df)s

    Returns
    -------
    %(count_positives)s
    %(count_negatives)s
    """
    assert 'diff' in df.columns

    # reset order
    df = df.sort_values(by=['participant', 'session', 'run', 'idx'],
                        ascending=True)
    df.reset_index()

    # check sign
    df['sign'] = np.sign(df['diff'])
    # groupby
    counts = df.groupby(by='participant', dropna=True)['sign'].value_counts()

    # create new counts dataframe
    positives = {'participant': [], 'count': []}
    negatives = {'participant': [], 'count': []}

    for participant in df['participant'].unique():
        try:
            pos = counts[participant, 1]
            neg = counts[participant, -1]
        except KeyError:
            pos = np.nan
            neg = np.nan

        # add common data to dict
        positives['participant'].append(participant)
        negatives['participant'].append(participant)

        if any(np.isnan(x) or x == 0 for x in (pos, neg)):
            positives['count'].append(np.nan)
            negatives['count'].append(np.nan)
        else:
            # normalize
            total = (pos + neg)
            pos = pos / total
            neg = neg / total
            # add to dict
            positives['count'].append(pos)
            negatives['count'].append(-neg)

    # create df
    df_positives = pd.DataFrame.from_dict(positives, orient='columns')
    df_negatives = pd.DataFrame.from_dict(negatives, orient='columns')

    # reset order
    df_positives.sort_values(by='participant', ascending=True, inplace=True)
    df_positives.reset_index()
    df_negatives.sort_values(by='participant', ascending=True, inplace=True)
    df_negatives.reset_index()

    return df_positives, df_negatives
