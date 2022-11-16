from neurotin.config import PARTICIPANTS
from neurotin.evamed.parsers import parse_thi
from neurotin.evamed.viz import lineplot_evolution
from neurotin.io import read_csv_evamed

#%% THI from multiple participants
fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/thi.csv"

df = read_csv_evamed(fname)
df = parse_thi(df, PARTICIPANTS)

f, ax = lineplot_evolution(df, "THI", figsize=(10, 5))
ax.grid(visible=True, which="major", axis="y")
