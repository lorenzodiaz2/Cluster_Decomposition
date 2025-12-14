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
        time_limit: int,
        output_flag: int
    ):
        self.G = G
        self.od_pairs = od_pairs
        self.m = None
        self.status = None
        self.time_limit = time_limit
        self.model_times = []
        self.resolution_times = []
        self.output_flag = output_flag

        self.A: List[Agent] = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.SP = {(od.src, od.dst): len(od.k_shortest_paths[0].visits) for od in self.od_pairs} ############# lunghezza totale, NON in passi



    def _optimize_model(self):
        start = time.perf_counter()
        self.m.Params.TimeLimit = self.time_limit
        self.m.optimize()
        self.resolution_times.append(time.perf_counter() - start)

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        self.status = status_map.get(self.m.Status, str(self.m.Status))
