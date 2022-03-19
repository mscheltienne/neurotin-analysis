from neurotin.io import read_csv_evamed
from neurotin.evamed.parsers import parse_thi
from neurotin.evamed.viz import lineplot_evolution


#%% THI from multiple participants
fname = r''
participants = []

df = read_csv_evamed(fname)
df = parse_thi(df, participants)

f, ax = lineplot_evolution(df, 'THI', figsize=(10, 5))
ax.grid(visible=True, which='major', axis='y')
