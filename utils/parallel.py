import numpy as np
import networkx as nx

from elements.pair import OD_Pair

# ----- per la matrice di similarità -----

_OD_PAIRS: list[OD_Pair] = []

def init_similarity_pool(od_pairs: list[OD_Pair]):
    global _OD_PAIRS
    _OD_PAIRS = od_pairs

def compute_similarity_row(i: int):
    n = len(_OD_PAIRS)
    row = np.zeros(n, dtype=int)
    od_i = _OD_PAIRS[i]
    for j in range(i + 1, n):
        od_j = _OD_PAIRS[j]
        row[j] = OD_Pair.compute_similarity(od_i, od_j)
    return i, row


# ----- per i k-shortest paths per quadrante -----

_GLOBAL_G: nx.Graph | None = None
_GLOBAL_K: int | None = None
_GLOBAL_RESTRICT_PATH_TO_QUADRANT: bool | None = False

def init_paths_pool(G: nx.Graph, k: int, restrict_paths_to_quadrant: bool):
    global _GLOBAL_G, _GLOBAL_K, _GLOBAL_RESTRICT_PATH_TO_QUADRANT
    _GLOBAL_G = G
    _GLOBAL_K = k
    _GLOBAL_RESTRICT_PATH_TO_QUADRANT = restrict_paths_to_quadrant

def _subgraph_for_quadrant_global(quadrant):
    (top, left), (bottom, right) = quadrant
    nodes = [(i, j) for i in range(top, bottom + 1) for j in range(left, right + 1)]
    return _GLOBAL_G.subgraph(nodes).copy()

def compute_paths_for_quadrant(args):
    quadrant, od_list = args
    sub_g = _subgraph_for_quadrant_global(quadrant) if _GLOBAL_RESTRICT_PATH_TO_QUADRANT else _GLOBAL_G
    for od in od_list:
        od.compute_k_shortest_paths(sub_g, _GLOBAL_K)
    return od_list
