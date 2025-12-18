import time
from functools import partial
from typing import List
import networkx as nx
from gurobipy import GRB, Model

from elements.agent import Agent
from elements.pair import OD_Pair


class Incumbent:
    def __init__(self):
        self.times = []
        self.solutions = []

def add_current_sol(model: Model, where, incumbent_obj):
    if where == GRB.Callback.MIPSOL:
        incumbent_obj.solutions.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        incumbent_obj.times.append(time.time() - model._start_time)


class General_Solver:
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: List[OD_Pair],
        time_limit: int,
        verbose: bool
    ):
        self.G = G
        self.od_pairs = od_pairs
        self.m = None
        self.status = None
        self.time_limit = time_limit
        self.model_times = []
        self.resolution_times = []
        self.output_flag = 1 if verbose else 0
        self.gap = None

        self.A: List[Agent] = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.SP = {(od.src, od.dst): len(od.k_shortest_paths[0].visits) for od in self.od_pairs} ############# lunghezza totale, NON in passi

        self.incumbent = Incumbent()



    def _optimize_model(self):
        start = time.perf_counter()
        self.m.Params.TimeLimit = self.time_limit
        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.m._start_time = time.time()
        self.m.optimize(callback)
        self.resolution_times.append(time.perf_counter() - start)

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        self.status = status_map.get(self.m.Status, str(self.m.Status))
        self.gap = self.m.MIPGap
