from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne import create_info
from mne.viz import plot_topomap

from neurotin.config import PARTICIPANTS
model_folder = Path('/Users/scheltie/Documents/datasets/neurotin/model/')

g1 = [68, 60, 57, 63, 75, 61, 83, 76]
g2 = [69, 72, 78, 79]
g3 = [81, 66, 65, 73]
PARTICIPANTS = g3

f, ax = plt.subplots(len(PARTICIPANTS), 4, figsize=(8, 20))

for i, run in enumerate(("full", "regular", "transfer")):
    fname = f'/Users/scheltie/Documents/datasets/neurotin/bandpower/delta-onrun-{run}-abs.pcl'
    df = pd.read_pickle(fname)

    ch_names = [
        col
        for col in df.columns
        if col not in ("participant", "session", "run", "idx")
    ]
    info = create_info(ch_names, 1, "eeg")
    info.set_montage("standard_1020")

    for k, participant in enumerate(PARTICIPANTS):
        df_ = df[df["participant"] == participant]
        df_ = df_[ch_names]
        data = np.average(df_.values, axis=0)
        plot_topomap(data - 1, info, axes=ax[k, i+1])

for k,participant in enumerate(PARTICIPANTS):
    model = model_folder / f"{str(participant).zfill(3)}.pcl"
    model = pd.read_pickle(model)
    info = create_info(list(model.index), 1, "eeg")
    info.set_montage("standard_1020")
    plot_topomap(model.values, info, axes=ax[k, 0])
    ax[k, 0].set_ylabel(str(participant).zfill(3))

ax[0, 0].set_title("Weights")
ax[0, 1].set_title("Full")
ax[0, 2].set_title("Regular")
ax[0, 3].set_title("Transfer")
f.tight_layout()
f.savefig("topo-delta-g3.svg")
