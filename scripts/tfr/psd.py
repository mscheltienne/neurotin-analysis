import pandas as pd

from neurotin.time_frequency.psd import (
    add_average_column, remove_outliers, diff_between_phases, ratio,
    count_diff)
from neurotin.time_frequency.plots import diff_catplot_distribution


#%% PSDs - Alpha
fname = r''

df_alpha = pd.read_pickle(fname)
df_alpha = add_average_column(df_alpha)
df_alpha = remove_outliers(df_alpha)
diff_alpha = diff_between_phases(df_alpha, column='avg')


#%% PSDs - Delta
fname = r''

df_delta = pd.read_pickle(fname)
df_delta = add_average_column(df_delta)
df_delta = remove_outliers(df_delta)
diff_delta = diff_between_phases(df_delta, column='avg')


#%% Ratio
df_ratio = ratio(df_alpha, df_delta)
diff_ratio = diff_between_phases(df_ratio, column='ratio')
df_positives, df_negatives = count_diff(diff_ratio)

participants = []
g = diff_catplot_distribution(df_positives, df_negatives, participants)
