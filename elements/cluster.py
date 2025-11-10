import random
from typing import List
from matplotlib.colors import to_hex
from elements.pair import OD_Pair

class Cluster:
    def __init__(self, cluster_id, od_pairs):
        self.id = cluster_id
        self.od_pairs: List[OD_Pair] = od_pairs
        self.n_agents = sum(len(od_pair.agents) for od_pair in self.od_pairs)
        self.color = to_hex((random.random(), random.random(), random.random()))

    def __str__(self) -> str:
        return f"Cluster {self.id}   number of agents = {self.n_agents}     pairs = {[pair.id for pair in self.od_pairs]}"

    def merge(self, other) -> None:
        self.od_pairs.extend(other.od_pairs)
        self.n_agents = sum(len(od_pair.agents) for od_pair in self.od_pairs)