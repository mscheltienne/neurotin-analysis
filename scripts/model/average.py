from pathlib import Path
import re

import pandas as pd

from neurotin.model import compute_average
from neurotin.model.viz import plot_topomap

#%% Folder
folder = r''
folder = Path(folder)

#%% Specific participants
participants = [63]
df = compute_average(folder, participants)

#%% All participants
pattern = re.compile(r'(\d{3})')
participants = [int(p.name) for p in folder.iterdir() if pattern.match(p.name)]
df = compute_average(folder, participants)

#%% Load dataframe
path = r''
df = pd.read_pickle(path)

#%% Plots
plot_topomap(df, outlines='head', contours=6, border=0, extrapolate='local')
