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


# ----- versione con indice inverso (utils/implementation.py, FAST_SIMILARITY) -----
# Valuta solo le coppie (i, j), j > i, che condividono almeno una risorsa: le altre hanno similarità 0 (nessun termine
# nella somma), come nella versione completa. Per le coppie valutate la funzione e gli argomenti sono gli stessi, quindi
# i valori sono identici bit per bit.

_RESOURCES_OF: list | None = None          # per ogni entità, le risorse che le sue opzioni possono usare
_ENTITIES_OF: dict | None = None           # per ogni risorsa, gli indici delle entità che possono usarla


def build_resource_index(elements, keys_fn: Callable[[Any], Any]) -> tuple[list, dict]:
    resources_of = [list(keys_fn(e)) for e in elements]
    entities_of: dict = {}
    for idx, keys in enumerate(resources_of):
        for key in keys:
            entities_of.setdefault(key, []).append(idx)
    return resources_of, entities_of


def init_similarity_indexed(elements, sim_fn: Callable[[Any, Any], float], resource_weights: dict | None,
                            resources_of: list, entities_of: dict):
    global _RESOURCES_OF, _ENTITIES_OF
    init_similarity(elements, sim_fn, resource_weights)
    _RESOURCES_OF = resources_of
    _ENTITIES_OF = entities_of


def compute_similarity_row_indexed(i: int):
    """Come compute_similarity_row, ma restituisce solo le colonne j > i con almeno una risorsa in comune."""
    candidates = set()
    for key in _RESOURCES_OF[i]:
        for j in _ENTITIES_OF[key]:
            if j > i:
                candidates.add(j)
    cols = np.fromiter(sorted(candidates), dtype=np.int64, count=len(candidates))
    element_i = _ELEMENTS[i]
    vals = np.fromiter((_SIM_FN(element_i, _ELEMENTS[j]) for j in cols), dtype=float, count=len(cols))
    return i, cols, vals
