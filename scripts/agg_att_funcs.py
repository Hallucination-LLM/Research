import numpy as np
from scipy.spatial.distance import jensenshannon

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

def dist_div_heads_agg(att_tensor: np.ndarray) -> np.ndarray:
    """
    Function to calculate the distance between the attention heads.
    """

    n_layers, n_heads, _, n_generated_tokens = att_tensor.shape

    att_tensor = np.concatenate((att_tensor, 1 - np.sum(att_tensor, axis=2, keepdims=True)), axis=2)
    reference_distribution = np.mean(att_tensor, axis=1)

    js_divergence = np.zeros((n_layers, n_heads, n_generated_tokens))

    for i in range(n_heads):
        js_divergence[:, i, :] = jensenshannon(att_tensor[:, i, ...], reference_distribution, axis=1)

    del att_tensor, reference_distribution
    js_divergence = np.mean(js_divergence, axis=-1)

    return js_divergence