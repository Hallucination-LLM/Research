import numpy as np

def weighted_avg_window_att(
    x: np.ndarray,
    window_size: int,
) -> float:
    
    # weights = -np.tanh(np.arange(1, window_size + 1) - window_size - 1)
    # weights /= weights.sum()

    return np.mean(x, axis=-1)

    # return np.average(x, weights=weights, axis=-1)

def agg_att_weighted(x: np.ndarray, window_size: int = 8, passage_percentage: float = 0.2) -> np.ndarray:

    return np.concatenate((
        weighted_avg_window_att(
            x=x.sum(axis=-1),
            window_size=window_size,
        ).flatten(), 
        np.array([passage_percentage])
    ))