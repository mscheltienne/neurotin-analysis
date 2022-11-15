from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from neurotin.config import PARTICIPANTS
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
f.savefig("/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate.svg")


#%% Subject learning rate
directory = Path(r"/Users/scheltie/Documents/datasets/neurotin/bandpower/")
df_alpha = pd.read_pickle(directory / "alpha-onrun-full-abs.pcl")
df_alpha = add_average_column(df_alpha)
df_delta = pd.read_pickle(directory / "delta-onrun-full-abs.pcl")
df_delta = add_average_column(df_delta)

for k, participant in enumerate(PARTICIPANTS):
    f, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)

    sns.boxplot(
        x="session",
        y="avg",
        data=df_alpha.loc[df_alpha["participant"] == participant],
        ax=ax[0],
        color="darkturquoise",
        fliersize=3.0,
    )
    sns.boxplot(
        x="session",
        y="avg",
        data=df_delta.loc[df_delta["participant"] == participant],
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
    ax[0].set_title(f"Learning rate - {str(participant).zfill(3)}")
    f.tight_layout()
    fname = f"{str(participant).zfill(3)}.svg"
    f.savefig(
        "/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate/" + fname
)


#%% Subject learning rate by groups
bad_learners = [60, 61, 63, 76, 81, 82, 83, 84, 85]
good_learners = [57, 65, 66, 68, 69, 72, 73, 75, 78, 79]
missing_data = [62, 77]

directory = Path(r"/Users/scheltie/Documents/datasets/neurotin/bandpower/")
df_alpha = pd.read_pickle(directory / "alpha-onrun-full-abs.pcl")
df_alpha = add_average_column(df_alpha)
df_delta = pd.read_pickle(directory / "delta-onrun-full-abs.pcl")
df_delta = add_average_column(df_delta)


f, ax = plt.subplots(
    len(bad_learners), 2, figsize=(8, 20), sharex=True, sharey=True
)
for k, participant in enumerate(bad_learners):
    for i, df in enumerate((df_alpha, df_delta)):
        df_participant = df.loc[df_alpha["participant"] == participant]
        x = [
            df_participant.loc[df_participant["session"] == ses]["avg"]
            for ses in range(1, 16, 1)
        ]
        x = [series for series in x if series.size != 0]
        positions = df_participant["session"].unique()
        ax[k, i].boxplot(
            x,
            positions=positions,
            patch_artist=True,
            boxprops=dict(
                facecolor="darkturquoise",
                zorder=.9,
                edgecolor="#3f3f3f",
                linewidth=1.5,
            ),
            whiskerprops=dict(color="#3f3f3f", linewidth=1.5, linestyle="-"),
            capprops=dict(color="#3f3f3f", linewidth=1.5),
            medianprops=dict(color="#3f3f3f", linewidth=1.5),
            flierprops=dict(
                markerfacecolor="#3f3f3f",
                marker="d",
                markeredgecolor="#3f3f3f",
                markersize=3.0,
            ),
        )
    ax[k, 0].set_ylabel("")
    ax[k, 0].set_ylabel(f"{str(participant).zfill(3)}")
    ax[k, 1].set_ylabel("")
for a in ax.flatten():
    a.set_ylim(bottom=0, top=3)
    a.axhline(y=1, color="black", linestyle="--", linewidth=1.0)
    a.set_xlabel("")
    a.set_xticks(ticks=np.arange(1., 16., 1.), labels=[str(k) for k in range(1, 16, 1)])
ax[0, 0].set_title("Average α (8, 13) Hz band power")
ax[0, 1].set_title("Average δ (1, 4) Hz band power")
ax[-1, 0].set_xlabel("Session n°")
ax[-1, 1].set_xlabel("Session n°")
f.tight_layout()
fname = "bad-regulators.svg"
f.savefig(
        "/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate/" + fname
)



f, ax = plt.subplots(
    len(good_learners), 2, figsize=(8, 20), sharex=True, sharey=True
)
for k, participant in enumerate(good_learners):
    for i, df in enumerate((df_alpha, df_delta)):
        df_participant = df.loc[df_alpha["participant"] == participant]
        x = [
            df_participant.loc[df_participant["session"] == ses]["avg"]
            for ses in range(1, 16, 1)
        ]
        x = [series for series in x if series.size != 0]
        positions = df_participant["session"].unique()
        ax[k, i].boxplot(
            x,
            positions=positions,
            patch_artist=True,
            boxprops=dict(
                facecolor="darkturquoise",
                zorder=.9,
                edgecolor="#3f3f3f",
                linewidth=1.5,
            ),
            whiskerprops=dict(color="#3f3f3f", linewidth=1.5, linestyle="-"),
            capprops=dict(color="#3f3f3f", linewidth=1.5),
            medianprops=dict(color="#3f3f3f", linewidth=1.5),
            flierprops=dict(
                markerfacecolor="#3f3f3f",
                marker="d",
                markeredgecolor="#3f3f3f",
                markersize=3.0,
            ),
        )
    ax[k, 0].set_ylabel("")
    ax[k, 0].set_ylabel(f"{str(participant).zfill(3)}")
    ax[k, 1].set_ylabel("")
for a in ax.flatten():
    a.set_ylim(bottom=0, top=3)
    a.axhline(y=1, color="black", linestyle="--", linewidth=1.0)
    a.set_xlabel("")
    a.set_xticks(ticks=np.arange(1., 16., 1.), labels=[str(k) for k in range(1, 16, 1)])
ax[0, 0].set_title("Average α (8, 13) Hz band power")
ax[0, 1].set_title("Average δ (1, 4) Hz band power")
ax[-1, 0].set_xlabel("Session n°")
ax[-1, 1].set_xlabel("Session n°")
f.tight_layout()
fname = "good-regulators.svg"
f.savefig(
        "/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate/" + fname
)
