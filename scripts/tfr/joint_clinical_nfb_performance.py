import pandas as pd

from neurotin.evamed.parsers import parse_thi
from neurotin.io import read_csv_evamed
from neurotin.time_frequency.psd import diff_between_phases, ratio
from neurotin.time_frequency.subject import count_diff
from neurotin.time_frequency.subject.viz import (
    plot_joint_clinical_nfb_performance)


#%% CLI commands
"""
# Compute PSDs
# -a accepts 'mean' or 'integrate'
# only difference is the scale, variations are identical -> integrate used.
neurotin_tfr_compute_psd_average_bins preprocessed/ica/ psds/alpha.pcl -p 57 60 61 63 65 66 68 72 73 -d 4 -o 2 --reject --fmin 8 --fmax 13 -a mean --n_jobs 35
neurotin_tfr_compute_psd_average_bins preprocessed/ica/ psds/delta.pcl -p 57 60 61 63 65 66 68 72 73 -d 4 -o 2 --reject --fmin 1 --fmax 4 -a mean --n_jobs 35

# Apply weights and remove outliers (python or IPython console)
import pandas as pd
from neurotin.time_frequency.psd import apply_weights_session

df = pd.read_pickle('psds/alpha.pcl')
df = apply_weights_session(df, 'data/Participants')
df = add_average_column(df)
df = remove_outliers(df, score=2.)
df.to_pickle('psds/alpha_.pcl', compression=None)
"""

#%% Participants
participants = []

#%% PSDs - Alpha
fname = r''
df_alpha = pd.read_pickle(fname)
diff_alpha = diff_between_phases(df_alpha, column='avg')


#%% PSDs - Delta
fname = r''
df_delta = pd.read_pickle(fname)
diff_delta = diff_between_phases(df_delta, column='avg')


#%% Ratio
df_ratio = ratio(df_alpha, df_delta)
diff_ratio = diff_between_phases(df_ratio, column='ratio')

#%% THI
fname = r''
df = read_csv_evamed(fname)
thi = parse_thi(df, participants)

#%% Plot
df = count_diff(diff_ratio)[0]
plot_joint_clinical_nfb_performance(df, thi, 'THI', participants)
