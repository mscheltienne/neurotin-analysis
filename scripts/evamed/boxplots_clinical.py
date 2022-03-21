from matplotlib import pyplot as plt
import seaborn as sns

from neurotin.evamed.parsers import (parse_thi, parse_stai, parse_bdi,
                                     parse_whodas, parse_psqi)
from neurotin.io import read_csv_evamed


# Set participants
participants = [57, 60, 61, 63, 65, 66, 68, 69, 72, 73]
# Load clinical dataframes
fname = r''
df = read_csv_evamed(fname)
thi = parse_thi(df, participants)
thi = thi[thi['visit'].isin(('Baseline', 'Post-assessment'))]

fname = r''
df = read_csv_evamed(fname)
stai = parse_stai(df, participants)

fname = r''
df = read_csv_evamed(fname)
bdi = parse_bdi(df, participants)

fname = r''
df = read_csv_evamed(fname)
psqi = parse_psqi(df, participants)

fname = r''
df = read_csv_evamed(fname)
whodas = parse_whodas(df, participants)

# Plots
f, ax = plt.subplots(1, 5, figsize=(20, 4))
order = ('Baseline', 'Post-assessment')
sns.boxplot(x='visit', y='result', data=thi, order=order, ax=ax[0],
            palette='muted')
sns.boxplot(x='visit', y='result', data=stai, order=order, ax=ax[1],
            palette='muted')
sns.boxplot(x='visit', y='result', data=bdi, order=order, ax=ax[2],
            palette='muted')
sns.boxplot(x='visit', y='result', data=psqi, order=order, ax=ax[3],
            palette='muted')
sns.boxplot(x='visit', y='result', data=whodas, order=order, ax=ax[4],
            palette='muted')

# Format
for ax_ in ax:
    ax_.set_ylabel('')
ax[0].set_ylabel('Score - lower the better')
for ax_ in ax:
    ax_.set_xlabel('')
ax[2].set_xlabel('Visit')

# Titles
ax[0].set_title('Tinnitus Handicap Inventory (THI)', fontsize=11)
ax[1].set_title('State-Trai Anxiety Inventory (STAI)', fontsize=11)
ax[2].set_title('Beck Depression Inventory (BDI)', fontsize=11)
ax[3].set_title('Pttsburgh Sleep Quality Index (PSQI)', fontsize=11)
ax[4].set_title('WHO Disability Assessment (WHODAS)', fontsize=11)

# Spacing
f.tight_layout()
f.subplots_adjust(left=0.05, right=0.95)
