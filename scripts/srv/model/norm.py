from os import makedirs

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from neurotin.config import PARTICIPANTS
from neurotin.config.srv import DATA_FOLDER, MODEL_FOLDER
from neurotin.model import (
    compute_weight_norm_per_session, create_weight_dataframe
)


makedirs(MODEL_FOLDER / "norm")
dfs = create_weight_dataframe(DATA_FOLDER, PARTICIPANTS)
for subject, df in dfs.items():
    sum_ = compute_weight_norm_per_session(df)
    f, ax = plt.subplots(1, 1)
    ax.hist(sum_, bins=15, density=True)
    ax.set_ylabel("Distribution of the L2 norm / session")
    ax.set_xlabel("L2 norm of the session weights")
    ax.set_title(f"Subject {subject}")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    f.tight_layout()
    f.savefig(MODEL_FOLDER / "norm" / f"dist-{str(subject).zfill(3)}.svg")
