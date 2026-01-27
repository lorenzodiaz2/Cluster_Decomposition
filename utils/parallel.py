import numpy as np
import networkx as nx
from typing import Callable, Any

from cfl.elements.facility import Facility


_GLOBAL_G: nx.Graph | None = None
_GLOBAL_K: int | None = None
_GLOBAL_FACILITIES: list[Facility] | None = None


def init_mcpa(G: nx.Graph, k: int):
    global _GLOBAL_G, _GLOBAL_K
    _GLOBAL_G = G
    _GLOBAL_K = k


# ----- per i k-shortest paths per quadrante -----


def compute_paths_for_quadrant(od_list):
    for od in od_list:
        od.compute_k_shortest_paths(_GLOBAL_G, _GLOBAL_K)
    return od_list



# ----- per i clients -----

def init_cfl(G: nx.Graph, k: int, facilities: list[Facility]):
    global _GLOBAL_G, _GLOBAL_K, _GLOBAL_FACILITIES
    _GLOBAL_G = G
    _GLOBAL_K = k
    _GLOBAL_FACILITIES = facilities


def find_facilities_for_clients(clients):
    for c in clients:
        c.find_nearest_facilities(_GLOBAL_G, _GLOBAL_FACILITIES, _GLOBAL_K)
    return clients




_ELEMENTS: list | None = None
_SIM_FN: Callable[[Any, Any], int] | None = None

def init_similarity(elements, sim_fn: Callable[[Any, Any], int]):
    global _ELEMENTS, _SIM_FN
    _ELEMENTS = elements
    _SIM_FN = sim_fn

def compute_similarity_row(i: int):
    n = len(_ELEMENTS)
    row = np.zeros(n, dtype=int)
    element_i = _ELEMENTS[i]
    for j in range(i + 1, n):
        row[j] = _SIM_FN(element_i, _ELEMENTS[j])
    return i, row



