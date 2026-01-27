import math
import networkx as nx
from cfl.elements.facility import Facility

class Client:
    def __init__(self, client_id, position, demand):
        self.id = client_id
        self.position = position
        self.demand = demand
        self.k_facilities: list[Facility] = []
        self.all_facilities: list[Facility] = []
        self.shipment_by_facility: dict[int, int] = {}


    def find_nearest_facilities(self, G: nx.Graph, facilities: list[Facility], k: int) -> None:
        dist = dict(nx.single_source_shortest_path_length(G, self.position))
        ordered = sorted(facilities, key=lambda fac: (dist.get(fac.position, math.inf), -fac.capacity))
        chosen: list[Facility] = []
        cap_sum = 0

        for f in ordered:
            chosen.append(f)
            cap_sum += f.capacity
            if len(chosen) >= k and cap_sum >= self.demand:
                break

        self.k_facilities = chosen
        self.all_facilities = ordered

    def add_facility(self) -> None:
        self.k_facilities.append(self.all_facilities[len(self.k_facilities)])

    def __str__(self):
        return f"Client: {self.id} position: {self.position} demand: {self.demand} facilities: {[f.id for f in self.k_facilities]}"

    def __eq__(self, other):
        return isinstance(other, Client) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


    @staticmethod
    def compute_similarity(c1, c2):



        ids1 = {f.id for f in c1.k_facilities}
        ids2 = {f.id for f in c2.k_facilities}
        return len(ids1 & ids2)








