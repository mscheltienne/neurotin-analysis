from neurotin.config import PARTICIPANTS
from neurotin.evamed.parsers import (
    parse_bdi,
    parse_psqi,
    parse_stai,
    parse_thi,
    parse_whodas,
)
from neurotin.evamed.viz import barplots_difference_between_visits
from neurotin.io import read_csv_evamed

# Set participants
participants = PARTICIPANTS
# Load clinical dataframes
fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/thi.csv"
df = read_csv_evamed(fname)
thi = parse_thi(df, participants)
thi = thi[thi["visit"].isin(("Baseline", "Post-assessment"))]

fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/stai.csv"
df = read_csv_evamed(fname)
stai = parse_stai(df, participants)

fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/bdi.csv"
df = read_csv_evamed(fname)
bdi = parse_bdi(df, participants)

fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/psqi.csv"
df = read_csv_evamed(fname)
psqi = parse_psqi(df, participants)

fname = r"/Users/scheltie/Documents/datasets/neurotin/evamed/whodas.csv"
df = read_csv_evamed(fname)
whodas = parse_whodas(df, participants)

# Plots
f, ax = barplots_difference_between_visits(
    [thi], ["thi"]
)
