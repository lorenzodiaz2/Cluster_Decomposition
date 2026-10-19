import numpy as np
from typing import Callable, Any


# ----- calcolo in parallelo della matrice di similarità (comune a tutti i problemi) -----

_ELEMENTS: list | None = None
_SIM_FN: Callable[[Any, Any], float] | None = None
_RESOURCE_WEIGHTS: dict | None = None

def init_similarity(elements, sim_fn: Callable[[Any, Any], float], resource_weights: dict | None = None):
    global _ELEMENTS, _SIM_FN, _RESOURCE_WEIGHTS
    _ELEMENTS = elements
    _SIM_FN = sim_fn
    _RESOURCE_WEIGHTS = resource_weights if resource_weights is not None else {}

def compute_similarity_row(i: int):
    n = len(_ELEMENTS)
    row = np.zeros(n, dtype=float)
    element_i = _ELEMENTS[i]
    for j in range(i + 1, n):
        row[j] = _SIM_FN(element_i, _ELEMENTS[j])
    return i, row
