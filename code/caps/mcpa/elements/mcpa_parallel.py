import networkx as nx


# ----- calcolo in parallelo dei path candidati delle coppie OD (variabili globali dei processi worker) -----

_GLOBAL_G: nx.Graph | None = None
_GLOBAL_K: int | None = None
_GLOBAL_SIM_METHOD: tuple[str, float | None] | None = ("uniform", None)


def init_mcpa(G: nx.Graph, k: int, sim_method: tuple[str, float | None]):
    global _GLOBAL_G, _GLOBAL_K, _GLOBAL_SIM_METHOD
    _GLOBAL_G = G
    _GLOBAL_K = k
    _GLOBAL_SIM_METHOD = sim_method


def compute_paths_for_pairs(od_list):
    for od in od_list:
        od.compute_k_shortest_paths(_GLOBAL_G, _GLOBAL_K, _GLOBAL_SIM_METHOD)
    return od_list
