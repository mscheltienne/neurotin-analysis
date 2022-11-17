import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from neurotin.evamed.parsers import parse_thi
from neurotin.io import read_csv_evamed
from neurotin.time_frequency import add_average_column
from neurotin.utils.align_axes import align_yaxis

PARTICIPANTS = [68, 60, 57, 63, 75, 61, 83, 76]

#%% Load dataframes
fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/thi.csv"
thi = parse_thi(read_csv_evamed(fname), PARTICIPANTS)

fname = '/Users/scheltie/Documents/datasets/neurotin/bandpower/alpha-onrun-full-abs.pcl'
df = pd.read_pickle(fname)
# df = df[df["session"].isin((1, 2, 3, 4, 5))]
# df = df[df["session"].isin((6, 7, 8, 9, 10))]
df = df[df["session"].isin((11, 12, 13, 14, 15))]

#%% Figure out the order by taking the diff in THI between post and baseline.
order = list()
for participant in PARTICIPANTS:
    df_ = thi[thi["participant"] == participant]
    baseline = int(df_[df_["visit"] == "Baseline"]["result"])
    post = int(df_[df_["visit"] == "Post-assessment"]["result"])
    order.append((participant, post - baseline))
order = [elt[0] for elt in sorted(order, key=lambda x: x[1])]

#%% Retrieve baseline and post values in the correct order
baselines = list()
post = list()
for participant in order:
    df_ = thi[thi["participant"] == participant]
    baselines.append(int(df_[df_["visit"] == "Baseline"]["result"]))
    post.append(int(df_[df_["visit"] == "Post-assessment"]["result"]))
baselines = np.array(baselines)
post = np.array(post)

#%% Count how many-times a participant managed to up-regulate the alpha-band
df = add_average_column(df)
up_regulations = dict()
down_regulations = dict()

# compute the count and the norm to estimate how strong the regulations are.
for participant in PARTICIPANTS:
    df_ = df[df["participant"] == participant]

    # up-regulations
    up_regulation = df_[df_["avg"] > 1]["avg"]
    up_regulations[participant] = (
        up_regulation.size / df_["avg"].size,
        np.linalg.norm(up_regulation),
    )

    # down-regulations
    down_regulation = df_[df_["avg"] < 1]["avg"]
    down_regulations[participant] = (
        down_regulation.size / df_["avg"].size,
        np.linalg.norm(1 / down_regulation),
    )

#%% Figure out the width limits
vmin = np.percentile(
    [elt[1] for elt in up_regulations.values()]
    + [elt[1] for elt in down_regulations.values()],
    10
)
vmax = np.percentile(
    [elt[1] for elt in up_regulations.values()]
    + [elt[1] for elt in down_regulations.values()],
    90
)
norm = Normalize(vmin, vmax, clip=True)

#%% Figure out the heights and the widths
up_regulations_heights = list()
up_regulation_widths = list()
down_regulation_heights = list()
down_regulation_widths = list()
for participant in order:
    up_regulations_heights.append(up_regulations[participant][0])
    width = norm(up_regulations[participant][1])
    up_regulation_widths.append(width)

    down_regulation_heights.append(down_regulations[participant][0])
    width = norm(down_regulations[participant][1])
    down_regulation_widths.append(width)

#%% Create figure
f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
x = np.arange(1, len(PARTICIPANTS) * 1.1, 1.1)
ax1.bar(
    x=x,
    height=up_regulations_heights,
    width=up_regulation_widths,
    align="center",
    facecolor="peru",
    edgecolor="#3f3f3f",
    label="α up-regulation",
)
ax1.bar(
    x=x,
    height=down_regulation_heights,
    width=down_regulation_widths,
    bottom=up_regulations_heights,
    align="center",
    facecolor="turquoise",
    edgecolor="#3f3f3f",
    label="α down-regulation",
)
ax1.set_xticks(x, labels=order)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.axhline(0.5, linestyle="--", color="black", linewidth=1)
ax1.yaxis.set_major_formatter(PercentFormatter(1))

ax2 = ax1.twinx()
ax2.bar(
    x=x - 0.5,
    height=-baselines,
    width=0.5,
    align="edge",
    facecolor="lightblue",
    edgecolor="#3f3f3f",
    label="baseline",
)
ax2.bar(
    x=x,
    height=-post,
    width=0.5,
    align="edge",
    facecolor="lightgreen",
    edgecolor="#3f3f3f",
    label="post-assessment",
)
align_yaxis(ax1, ax2)
ax2.set_yticks(
    np.arange(0, -100, -20), [str(abs(k)) for k in np.arange(0, -100, -20)]
)

ax1.set_xlabel("Participant n°")
ax1.set_ylabel("Number of up and down regulations")
ax2.set_ylabel("THI score (baseline, post-assessment)")
ax1.legend(loc="upper right")
ax2.legend(loc="lower right")
ax1.set_title("Session 10 to 15")
f.tight_layout()
f.savefig("good-ses-10-15.svg")
