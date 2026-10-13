import math
import networkx as nx
from caps.cfl.elements.facility import Facility
from utils import parallel


class Client:
    def __init__(self, client_id, position, demand):
        self.id = client_id
        self.position = position
        self.demand = demand
        self.k_facilities: list[Facility] = []
        self.all_facilities: list[Facility] = []
        self.shipment_by_facility: dict[int, int] = {}
        self.distances_to_facilities: dict[int, float] = {}
        self.option_weights = {}


    def find_nearest_facilities(
            self,
            G: nx.Graph,
            facilities: list[Facility],
            k: int,
            weight: str | None = None
    ) -> None:
        if weight is None:
            dist_to_nodes = dict(nx.single_source_shortest_path_length(G, self.position))
        else:
            dist_to_nodes = dict(nx.single_source_dijkstra_path_length(G, self.position, weight=weight))

        self.distances_to_facilities = {
            fac.id: dist_to_nodes.get(fac.position, math.inf)
            for fac in facilities
        }

        ordered = sorted(
            facilities,
            key=lambda fac: (self.distances_to_facilities.get(fac.id, math.inf), -fac.capacity),
        )

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


    def build_option_weights(self, weighting: str, beta: float | None):
        raw_weights = {}
        min_dist = min((self.distances_to_facilities[f.id] for f in self.k_facilities), default=0)

        for f in self.k_facilities:
            c = self.distances_to_facilities[f.id] - min_dist

            if weighting == "uniform":
                w = 1.0
            elif weighting == "inverse":
                w = 1.0 / (c + 1.0)
            elif weighting == "exponential":
                w = math.exp(-beta * c) if beta else math.exp(-c)
            else:
                raise ValueError(f"Weighting '{weighting}' non supportato")

            raw_weights[f.id] = w

        total_weight = sum(raw_weights.values())
        self.option_weights = {f_id: w / total_weight for f_id, w in raw_weights.items()}



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
        shared_ids = ids1 & ids2

        if not shared_ids:
            return 0.0

        rw = parallel._RESOURCE_WEIGHTS or {}

        min_demand = min(float(c1.demand), float(c2.demand))

        sim = 0.0
        for f_id in shared_ids:
            w_r = rw.get(f_id, 1.0)  # w(r): peso lineare/esponenziale o 1 se uniform
            # La formula dell'overlap: w(r) * a_e(r) * a_e'(r)
            sim += c1.option_weights[f_id] * c2.option_weights[f_id] * w_r

        return min_demand * sim

