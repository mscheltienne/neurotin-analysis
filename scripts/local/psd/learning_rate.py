from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from neurotin.time_frequency import add_average_column

#%% Group learning rate
directory = Path(r"/Users/scheltie/Documents/datasets/neurotin/bandpower/")
df_alpha = pd.read_pickle(directory / "alpha-onrun-full-abs.pcl")
df_alpha = add_average_column(df_alpha)
df_delta = pd.read_pickle(directory / "delta-onrun-full-abs.pcl")
df_delta = add_average_column(df_delta)

f, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
sns.boxplot(
    x="session",
    y="avg",
    data=df_alpha,
    ax=ax[0],
    color="darkturquoise",
    fliersize=3.0,
)
sns.boxplot(
    x="session",
    y="avg",
    data=df_delta,
    ax=ax[1],
    color="darkturquoise",
    fliersize=3.0,
)
for a in ax:
    a.set_ylim(bottom=0, top=3)
    a.axhline(y=1, color="black", linestyle="--", linewidth=1.0)
ax[0].set_xlabel("")
ax[1].set_xlabel("Session n°")
ax[0].set_ylabel("Average α (8, 13) Hz bandpower")
ax[1].set_ylabel("Average δ (1, 4) Hz bandpower")
ax[0].set_title("Group learning rate")
f.tight_layout()
