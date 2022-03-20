import pandas as pd

from neurotin.evamed.parsers import parse_thi
from neurotin.io import read_csv_evamed
from neurotin.psd import (
    blocks_difference_between_consecutive_phases, ratio, blocks_count_success)
from neurotin.psd.viz import plot_joint_clinical_nfb_performance


#%% CLI commands
"""
# Compute PSDs
# -a accepts 'mean' or 'integrate'
# only difference is the scale, variations are identical -> integrate used.
neurotin_psd_avg_band preprocessed/ica/ psds/alpha.pcl -p 57 60 61 63 65 66 68 72 73 -d 4 -o 2 --reject --fmin 8 --fmax 13 -a mean --n_jobs 35
neurotin_psd_avg_band preprocessed/ica/ psds/delta.pcl -p 57 60 61 63 65 66 68 72 73 -d 4 -o 2 --reject --fmin 1 --fmax 4 -a mean --n_jobs 35

# Apply weights and remove outliers (python or IPython console)
import pandas as pd
from neurotin.psd import weights_apply_session_mask, add_average_column, remove_outliers

df = pd.read_pickle('psds/alpha.pcl')
df = weights_apply_session_mask(df, 'data/Participants')
df = add_average_column(df)
df = remove_outliers(df, score=2.)
df.to_pickle('psds/alpha_.pcl', compression=None)
"""

#%% Participants
participants = []

#%% PSDs - Alpha
fname = r''
df_alpha = pd.read_pickle(fname)
diff_alpha = blocks_difference_between_consecutive_phases(
    df_alpha, column='avg')

#%% PSDs - Delta
fname = r''
df_delta = pd.read_pickle(fname)
diff_delta = blocks_difference_between_consecutive_phases(
    df_delta, column='avg')

#%% Ratio
df_ratio = ratio(df_alpha, df_delta)
diff_ratio = blocks_difference_between_consecutive_phases(
    df_ratio, column='ratio')

#%% THI
fname = r''
df = read_csv_evamed(fname)
thi = parse_thi(df, participants)

#%% Plot
df_positives, _ = blocks_count_success(diff_ratio, group_session=True)
plot_joint_clinical_nfb_performance(df_positives, thi, 'THI', participants)
