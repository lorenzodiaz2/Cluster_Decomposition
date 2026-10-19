import networkx as nx

from caps.cfl.elements.facility import Facility


# ----- ricerca in parallelo delle facility più vicine ai clienti (variabili globali dei processi worker) -----

_GLOBAL_G: nx.Graph | None = None
_GLOBAL_K: int | None = None
_GLOBAL_FACILITIES: list[Facility] | None = None


def init_cfl(G: nx.Graph, k: int, facilities: list[Facility]):
    global _GLOBAL_G, _GLOBAL_K, _GLOBAL_FACILITIES
    _GLOBAL_G = G
    _GLOBAL_K = k
    _GLOBAL_FACILITIES = facilities


def find_facilities_for_clients(clients):
    for c in clients:
        c.find_nearest_facilities(_GLOBAL_G, _GLOBAL_FACILITIES, _GLOBAL_K)
    return clients
