from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from neurotin.config import PARTICIPANTS
from neurotin.time_frequency import add_average_column

#%% Load dataframes
fname = '/Users/scheltie/Documents/datasets/neurotin/bandpower/alpha-onrun-full-abs.pcl'
df = pd.read_pickle(fname)
df = add_average_column(df)

#%% Create figures
linear_fits = dict()
for participant in PARTICIPANTS:
    df_ = df[df["participant"] == participant]
    df_ = df_.sort_values(by=["session", "run", "idx"], ignore_index=True)

    # retrieve data
    data = list()
    for session in range(1, 16):
        for run in range(1, 7 if session != 1 else 6):
            sel = df_[(df_["session"] == session) & (df_["run"] == run)]
            if sel.size == 0:
                data.extend([np.nan] * 10)
            else:
                data.extend(sel["avg"])

    # linear fit
    X = [k for k, elt in enumerate(data) if not np.isnan(elt)]
    Y = [elt for elt in data if not np.isnan(elt)]
    z = np.polyfit(X, Y, 1)
    linear_fits[participant] = z

    # plot
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(data)
    ax.plot(
        z[0] * np.arange(len(data)) + z[1],
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 3)
    ax.set_title(f"Participant {str(participant).zfill(3)}")

    # annotate session
    colors = cycle(("green", "blue"))
    x = np.arange(0, 50, 1)  # session 1
    ax.fill_between(x, 0, 3, facecolor=next(colors), alpha=0.2)
    for k in range(14):  # session 2 to 15
        x = np.arange(50 + k * 60, 50 + (k+1) * 60, 1)
        ax.fill_between(x, 0, 3, facecolor=next(colors), alpha=0.2)

    # figure out the x-ticks location
    x = [25] + list(range(80, 861, 60))
    ax.set_xticks(x, labels=range(1, 16))
    ax.set_xlabel("Session n°")
    ax.set_ylabel("regulation / rest α band power")
    f.tight_layout()
    f.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate/linear fits/alpha/' + str(participant).zfill(3) + ".svg")

#%% Plot the linear fits ax + b
plt.close("all")

# 62 and 77 are dropped
f, ax = plt.subplots(1, 1, figsize=(10, 5))
g1 = [68, 60, 57, 63, 75, 61, 83, 76]
g2 = [69, 72, 82, 78, 79]
g3 = [85, 81, 66, 65, 73, 84]
g3 = [85, 81, 66]
order = g1 + g2 + g3

# prepare data
x = np.arange(1, len(order)+1)
coeff = np.array([linear_fits[p][0] for p in order])
intercept = np.array([linear_fits[p][1] for p in order])

# normalize both vectors
norm = np.linalg.norm(intercept)
intercept = intercept / norm
coeff = coeff / np.linalg.norm(coeff)

ax.axhline(1 / norm, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x, order)

for k, participant in enumerate(order):
    ax.arrow(
        x=x[k], y=intercept[k] + 0.0025, dx=0, dy=coeff[k],
        width=0.001,
        length_includes_head=True,
        head_width=0.2,
        head_length=0.01,
        facecolor="#3f3f3f",
        edgecolor="#3f3f3f",
    )
ax.bar(
    x, height=0.005, width=0.3, bottom=intercept,
    facecolor="turquoise", edgecolor="#3f3f3f",
)
ax.set_xlabel("Participant n°")
ax.set_ylabel("Normalize learning rate")
ax.set_ylim(0, 1.3)
f.tight_layout()
f.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/learning-rate/linear fits/alpha/g123.svg')
