import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from neurotin.config import PARTICIPANTS
from neurotin.evamed.parsers import parse_thi
from neurotin.io import read_csv_evamed
from neurotin.time_frequency import add_average_column

PARTICIPANTS = [elt for elt in PARTICIPANTS if elt not in (77,)]


#%% Load dataframes
fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/thi.csv"
thi = parse_thi(read_csv_evamed(fname), PARTICIPANTS)
thi["result"] = -thi["result"]

fname = '/Users/scheltie/Documents/datasets/neurotin/bandpower/alpha-onrun-transfer-abs.pcl'
df = pd.read_pickle(fname)

#%% Figure out the order by taking the diff in THI between post and baseline.
order = list()
for participant in PARTICIPANTS:
    df_ = thi[thi["participant"] == participant]
    baseline = int(df_[df_["visit"] == "Baseline"]["result"])
    post = int(df_[df_["visit"] == "Post-assessment"]["result"])
    order.append((participant, post - baseline))
order = [elt[0] for elt in sorted(order, key=lambda x: x[1], reverse=True)]

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
        np.linalg.norm(up_regulation - 1),
    )

    # down-regulations
    down_regulation = df_[df_["avg"] < 1]["avg"]
    down_regulations[participant] = (
        down_regulation.size / df_["avg"].size,
        np.linalg.norm(1 - down_regulation),
    )

#%% Figure out the width limits
vmin = min(
    np.percentile([elt[1] for elt in up_regulations.values()], 0),
    np.percentile([elt[1] for elt in down_regulations.values()], 0)
)
vmax = max(
    np.percentile([elt[1] for elt in up_regulations.values()], 90),
    np.percentile([elt[1] for elt in down_regulations.values()], 90)
)

#%% Figure out the heights and the widths
up_regulations_heights = list()
up_regulation_widths = list()
down_regulation_heights = list()
down_regulation_widths = list()
for participant in order:
    up_regulations_heights.append(up_regulations[participant][0])
    width = (up_regulations[participant][1] - vmin) / (vmax - vmin)
    up_regulation_widths.append(np.clip(width, 0.1, 1))

    down_regulation_heights.append(down_regulations[participant][0])
    width = (down_regulations[participant][1] - vmin) / (vmax - vmin)
    down_regulation_widths.append(np.clip(width, 0.1, 1))

#%% Create figure
f, ax = plt.subplots(1, 1, figsize=(10, 5))
x = np.arange(1, len(PARTICIPANTS) * 1.1, 1.1)
ax.bar(
    x=x,
    height=up_regulations_heights,
    width=up_regulation_widths,
    align="center",
    facecolor="peru",
    edgecolor="#3f3f3f",
)
ax.bar(
    x=x,
    height=down_regulation_heights,
    width=down_regulation_widths,
    bottom=up_regulations_heights,
    align="center",
    facecolor="lightblue",
    edgecolor="#3f3f3f",
)
ax.set_xticks(x, labels=order)
ax.yaxis.set_major_formatter(PercentFormatter(1))
