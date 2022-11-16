import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_weight_norm_per_session(df: pd.DataFrame) -> NDArray[float]:
    """Compute the distribution of the weight L2 norm per session."""
    data = np.nan_to_num(df.values, nan=0)
    return np.linalg.norm(data, axis=0)
