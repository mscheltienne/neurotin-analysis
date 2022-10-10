import os
from pathlib import Path

from neurotin.config import participants
from neurotin.model import compute_average
from neurotin.model.viz import plot_topomap


data_folder = Path(r"/media/miplab-nas2/Data3/NeuroTinEEG/data/Participants/")
model_folder = Path(r"/media/miplab-nas2/Data3/NeuroTinEEG/model/")
# create folders
os.makedirs(model_folder, exist_ok=True)
os.makedirs(model_folder / "viz.eeglab", exist_ok=True)
os.makedirs(model_folder / "viz.mne", exist_ok=True)

# group-level average model
df = compute_average(data_folder, participants)
df.to_pickle(model_folder / "avg.pcl")

ax, _ = plot_topomap(df, show=False)
ax.figure.savefig(model_folder / "viz.mne" / "avg.svg")
ax.figure.clf()
ax, _ = plot_topomap(df, sphere="eeglab", show=False)
ax.figure.savefig(model_folder / "viz.eeglab" / "avg.svg")
ax.figure.clf()

# subject-level average model
for participant in participants:
    df = compute_average(data_folder, participant)
    df.to_pickle(model_folder / f"{str(participant).zfill(3)}.pcl")

    ax, _ = plot_topomap(df, show=False)
    ax.figure.savefig(
        model_folder / "viz.mne" / f"{str(participant).zfill(3)}.svg"
    )
    ax.figure.clf()
    ax, _ = plot_topomap(df, sphere="eeglab", show=False)
    ax.figure.savefig(
        model_folder / "viz.eeglab" / f"{str(participant).zfill(3)}.svg"
    )
    ax.figure.clf()
