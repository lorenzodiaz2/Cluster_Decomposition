from collections import defaultdict
from typing import List
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from elements.pair import OD_Pair
from elements.path import Path
from solvers.general_solver import General_Solver
from solvers.post_processing import Critical_Resources


class Heuristic_Solver(General_Solver):
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: List[OD_Pair],
        T: int
    ):
        super().__init__(G, od_pairs, T)

        self.P = {}
        self.K = {}
        self.E = {}
        self.x = None
        self.z = None
        self.critical_resources = None
        self.lb = {}


    def solve(
        self,
        critical_resources: Critical_Resources | None = None
    ):
        if critical_resources:
            self.critical_resources = critical_resources

        self.set_model()
        self.optimize_model()
        while self.status != "OPTIMAL":
            if self.critical_resources:
                # new_tol = self.critical_resources.tol + 1
                # self.critical_resources = Critical_Resources(self.G, self.od_pairs, new_tol)
                self.critical_resources.increment_tol()
                self.critical_resources.unassign_agents()
            else:
                self.T += 1
                for od_pair in self.od_pairs:
                    od_pair.delay_shortest_paths(self.T)
            self.set_model()
            self.optimize_model()
        self.assign_solutions()


    def set_model(self):
        self.P = {(od_pair, j): p for od_pair in self.od_pairs for j, p in enumerate(od_pair.all_paths)}
        self.K = defaultdict(list) # K[od] = {1, 2, ...}
        self.E = defaultdict(list)

        for od_pair, j in self.P.keys():
            self.K[od_pair].append(j)
            for t, v in enumerate(self.P[od_pair, j].visits):
                self.E[v, t].append((od_pair, j))

        self.m = gp.Model("heuristic")
        self.m.Params.OutputFlag = 0

        keys = [(od.id, j) for (od, j) in self.P.keys()]
        self.compute_lb()
        self.x = self.m.addVars(keys, vtype=GRB.INTEGER, lb=self.lb)
        self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for j in self.K[od_pair]) == len(od_pair.agents) for od_pair in self.od_pairs)
        self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for (od_pair, j) in self.E[v, t]) <= self.G.nodes[v]["capacity"] for (v, t) in self.E.keys())

        self.m.setObjective(gp.quicksum(self.x[od_pair.id, j] * (len(self.P[od_pair, j].visits) - 1 - self.SP[od_pair.src, od_pair.dst]) for od_pair, j in self.P.keys()), GRB.MINIMIZE)


    def compute_lb(self):
        if self.critical_resources:
            removed = set(self.critical_resources.removed_agents)
            keys = []
            path_to_j = {}
            lb = {}

            for (od, j), p in self.P.items():
                keys.append((od.id, j))
                path_to_j[od, tuple(p.visits)] = j
                lb[(od.id, j)] = 0

            self.lb = {(od.id, path_to_j[od, tuple(a.path.visits)]): 0 if a in removed else 1 for od in self.od_pairs for a in od.agents}
        else:
            self.lb = 0


    def assign_solutions(self):
        if self.status == "OPTIMAL" and self.m.SolCount > 0:
            non_assigned_agents = {od_pair.id: [a for a in od_pair.agents] for od_pair in self.od_pairs}
            for (od_pair, j), p in self.P.items():
                n_agents = int(self.x[od_pair.id, j].X)
                if n_agents > 0:
                    delay = len(p.visits) - 1 - self.SP[od_pair.src, od_pair.dst]
                    for i in range(n_agents):
                        a = non_assigned_agents[od_pair.id][i]
                        a.path = Path(list(p.visits))
                        a.delay = delay
                    del non_assigned_agents[od_pair.id][:n_agents]
