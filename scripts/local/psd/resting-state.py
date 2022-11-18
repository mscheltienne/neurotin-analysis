import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from neurotin.time_frequency import add_average_column


#%% group level
fname = '/Users/scheltie/Documents/datasets/neurotin/alpha-rs-rel.pcl'
df_alpha = pd.read_pickle(fname)
df_alpha = add_average_column(df_alpha)
fname = '/Users/scheltie/Documents/datasets/neurotin/delta-rs-rel.pcl'
df_delta = pd.read_pickle(fname)
df_delta = add_average_column(df_delta)

f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
sns.boxplot(x="session", y="avg", data=df_alpha, palette="muted", ax=ax[0])
sns.boxplot(x="session", y="avg", data=df_delta, palette="muted", ax=ax[1])
ax[0].set_ylabel("Average relative band power")
ax[1].set_ylabel("")
ax[0].set_xlabel("Session n°")
ax[1].set_xlabel("Session n°")
ax[0].set_title("α (8, 13) Hz")
ax[1].set_title("δ (1, 4) Hz")
f.tight_layout()
f.savefig("group-rs.svg")

#%% sub-group level
f, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10, 15))
g1 = [68, 60, 57, 63, 75, 61, 83, 76]
g2 = [69, 72, 78, 79]
g3 = [81, 66, 65, 73]
for k, group in enumerate((g1, g2, g3)):
    df_alpha_ = df_alpha[df_alpha["participant"].isin(group)]
    df_delta_ = df_delta[df_delta["participant"].isin(group)]
    sns.boxplot(x="session", y="avg", data=df_alpha_, palette="muted", ax=ax[k, 0])
    sns.boxplot(x="session", y="avg", data=df_delta_, palette="muted", ax=ax[k, 1])
    ax[k, 0].set_xlabel("")
    ax[k, 1].set_xlabel("")
    ax[k, 0].set_ylabel("")
    ax[k, 1].set_ylabel("")
ax[-1, 0].set_xlabel("Session n°")
ax[-1, 1].set_xlabel("Session n°")
ax[1, 0].set_ylabel("Average relative band power")
ax[0, 0].set_title("α (8, 13) Hz")
ax[0, 1].set_title("δ (1, 4) Hz")
f.tight_layout()
f.savefig("subgroup-rs.svg")
