

import numpy as np





def explained_variance(g: np.ndarray, v: np.ndarray) -> float:
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    var_g = float(np.var(g))
    if var_g < 1e-12:
        return float("nan")
    return float(1.0 - np.var(g - v) / var_g)
