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
        T: int
    ):
        self.G = G
        self.od_pairs = od_pairs
        self.T = T
        self.m = None
        self.status = None

        self.A: List[Agent] = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.SP = {(od.src, od.dst): nx.shortest_path_length(self.G, od.src, od.dst) for od in self.od_pairs}



    def optimize_model(self):
        self.m.optimize()

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        self.status = status_map.get(self.m.Status, str(self.m.Status))
        # print(f"T = {self.T} -> {self.status}")
