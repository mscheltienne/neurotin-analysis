from os import makedirs

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, MODEL_FOLDER
from neurotin.model import compute_average
from neurotin.model.viz import plot_topomap


# create folders
makedirs(MODEL_FOLDER, exist_ok=True)
makedirs(MODEL_FOLDER / "viz.eeglab", exist_ok=True)
makedirs(MODEL_FOLDER / "viz.mne", exist_ok=True)

# group-level average model
df = compute_average(DATA_FOLDER, PARTICIPANTS)
df.to_pickle(MODEL_FOLDER / "avg.pcl")

ax, _ = plot_topomap(df, show=False)
ax.figure.savefig(MODEL_FOLDER / "viz.mne" / "avg.svg")
ax.figure.clf()
ax, _ = plot_topomap(df, sphere="eeglab", show=False)
ax.figure.savefig(MODEL_FOLDER / "viz.eeglab" / "avg.svg")
ax.figure.clf()

# subject-level average model
for participant in PARTICIPANTS:
    df = compute_average(DATA_FOLDER, participant)
    df.to_pickle(MODEL_FOLDER / f"{str(participant).zfill(3)}.pcl")

    ax, _ = plot_topomap(df, show=False)
    ax.figure.savefig(
        MODEL_FOLDER / "viz.mne" / f"{str(participant).zfill(3)}.svg"
    )
    ax.figure.clf()
    ax, _ = plot_topomap(df, sphere="eeglab", show=False)
    ax.figure.savefig(
        MODEL_FOLDER / "viz.eeglab" / f"{str(participant).zfill(3)}.svg"
    )
    ax.figure.clf()
