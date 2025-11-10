import time
from typing import List

import networkx as nx
from gurobipy import GRB

from elements.agent import Agent
from elements.pair import OD_Pair


class General_Solver:
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: List[OD_Pair],
        T: int,
        max_time: int
    ):
        self.G = G
        self.od_pairs = od_pairs
        self.starting_T = T
        self.T = T
        self.m = None
        self.status = None
        self.max_time = max_time
        self.model_times = []
        self.resolution_times = []

        self.A: List[Agent] = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.SP = {(od.src, od.dst): nx.shortest_path_length(self.G, od.src, od.dst) for od in self.od_pairs}



    def optimize_model(self):
        start = time.time()
        self.m.Params.TimeLimit = self.max_time
        self.m.optimize()

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        self.status = status_map.get(self.m.Status, str(self.m.Status))
        self.resolution_times.append(time.time() - start)
