from neurotin.io import read_csv_evamed
from neurotin.evamed.parsers import parse_multi_thi
from neurotin.evamed.thi import plot_multi_thi_evolution


#%% THI from multiple participants
fname = r''
participants = []

df = read_csv_evamed(fname)
df = parse_multi_thi(df, participants)

f, ax = plot_multi_thi_evolution(df, figsize=(10, 5))
ax.grid(visible=True, which='major', axis='y')
