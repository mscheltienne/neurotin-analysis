import pandas as pd

from .average import add_average_column


def ratio(df_alpha, df_delta):
    """Compute the ratio of alpha/delta band power.

    Parameters
    ----------
    df_alpha : DataFrame
        Bandpower in the alpha band.
    df_delta : DataFrame
        Bandpower in the delta band.

    Returns
    -------
    df : DataFrame
    """
    if "avg" not in df_alpha.columns:
        df_alpha = add_average_column(df_alpha)
    if "avg" not in df_delta.columns:
        df_alpha = add_average_column(df_delta)

    # check keys
    keys = ["participant", "session", "run", "phase", "idx"]
    assert len(set(keys).intersection(df_alpha.columns)) == len(keys)
    assert len(set(keys).intersection(df_delta.columns)) == len(keys)
    assert sorted(df_alpha.columns) == sorted(df_delta.columns)

    # container for new df with ratio of power
    data = {key: [] for key in keys + ["ratio"]}

    ratio = df_alpha["avg"] / df_delta["avg"]
    ratio = ratio[ratio.notna()]

    # fill new df dict
    for i, r in ratio.iteritems():
        alpha_ = df_alpha.loc[i]
        delta_ = df_delta.loc[i]

        # sanity-check
        try:
            assert alpha_["participant"] == delta_["participant"]
            assert alpha_["session"] == delta_["session"]
            assert alpha_["run"] == delta_["run"]
            assert alpha_["phase"] == delta_["phase"]
            assert alpha_["idx"] == delta_["idx"]
        except AssertionError:
            continue

        data["participant"].append(alpha_["participant"])
        data["session"].append(alpha_["session"])
        data["run"].append(alpha_["run"])
        data["phase"].append(alpha_["phase"])
        data["idx"].append(alpha_["idx"])
        data["ratio"].append(r)

    # create df
    df = pd.DataFrame.from_dict(data, orient="columns")
    return df
