import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne import create_info
from mne.viz import plot_topomap

from neurotin.psd import blocks_difference_between_consecutive_phases

#%% Participant
participants = []

#%% Load
fname = r""
df = pd.read_pickle(fname)
diff = blocks_difference_between_consecutive_phases(df, column="all")

#%% Create plots
f, ax = plt.subplots(2, len(participants), figsize=(20, 4))

#%% Compute and plot topographic maps
electrodes = [
    col
    for col in diff.columns
    if col not in ("participant", "session", "run", "idx", "avg-diff")
]
info = create_info([elt.split("-")[0] for elt in electrodes], 1, "eeg")
info.set_montage("standard_1020")

for i, participant in enumerate(participants):
    upreg = {k: np.zeros(len(electrodes)) for k in diff.session.unique()}
    downreg = {k: np.zeros(len(electrodes)) for k in diff.session.unique()}

    df_participant = diff[diff["participant"] == participant]
    sessions = sorted(df_participant["session"].unique())
    for session in sessions:
        df_session = df_participant[df_participant["session"] == session]
        data = df_session[electrodes].values
        pos = np.argwhere(data > 0)
        neg = np.argwhere(data < 0)
        for idx in pos:
            upreg[session][idx[1]] += data[idx[0], idx[1]]
        for idx in neg:
            downreg[session][idx[1]] += data[idx[0], idx[1]]

    upreg_ = np.average(np.stack(list(upreg.values())), axis=0)
    downreg_ = np.average(np.stack(list(downreg.values())), axis=0)

    # plot
    plot_topomap(upreg_, info, axes=ax[0, i], extrapolate="local", show=False)
    plot_topomap(
        downreg_, info, axes=ax[1, i], extrapolate="local", show=False
    )
    ax[0, i].set_title(str(participant))
    ax[0, 0].set_ylabel("Up-regulation")
    ax[1, 0].set_ylabel("Down-regulation")

f.tight_layout()
