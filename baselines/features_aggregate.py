import numpy as np


AGG_FUNCS = [np.mean, np.std, lambda x: x[-1], lambda x: (x[-1]-x[0])/(len(x))]


def make_aggregates(window: np.ndarray):
    # window: (T, F) -> hand-crafted features per channel
    feats = []
    for f in range(window.shape[1]):
        x = window[:, f]
        feats.extend([fn(x) for fn in AGG_FUNCS])
    return np.array(feats)