import time
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
        T: int,
        critical_resources: Critical_Resources | None = None
    ):
        super().__init__(G, od_pairs, T)

        self.P = None
        self.K = None
        self.E = None
        self.x = None
        self.critical_resources = critical_resources



    def solve(self):
        self._set_model()
        self._optimize_model()
        while self.status == "INFEASIBLE":
            if self.critical_resources:
                print("  (tol+1)  ")
                self.critical_resources.increment_tol()
                self.critical_resources.unassign_agents()
            else:
                self.current_T += 1
                for od_pair in self.od_pairs:
                    od_pair.delay_shortest_paths(self.current_T)
            self._set_model()
            self._optimize_model()


    def _set_model(self):
        start = time.time()

        self.m = gp.Model("heuristic")
        self.m.Params.OutputFlag = 0

        self._set_variables()
        self._set_constraints()

        self.m.setObjective(gp.quicksum(self.x[od_pair.id, j] * (len(p.visits) - self.SP[od_pair.src, od_pair.dst]) for (od_pair, j), p in self.P.items()), GRB.MINIMIZE)
        self.model_times.append(time.time() - start)



    def _set_variables(self):
        if self.critical_resources:
            self.P = {(od_pair, j): p for od_pair in self.critical_resources.critical_od_pairs for j, p in enumerate(od_pair.all_paths) if all(self.critical_resources.left_caps.get((v, t), self.G.nodes[v]["capacity"]) > 0 for t, v in enumerate(p.visits))}
        else:
            self.P = {(od_pair, j): p for od_pair in self.od_pairs for j, p in enumerate(od_pair.all_paths)}

        self.K = defaultdict(list) # K[od] = {1, 2, ...}
        self.E = defaultdict(list) # E[v, t] = [(od, j), ...]

        for od_pair, j in self.P.keys():
            self.K[od_pair].append(j)
            for t, v in enumerate(self.P[od_pair, j].visits):
                self.E[v, t].append((od_pair, j))

        self.x = self.m.addVars([(od.id, j) for (od, j) in self.P.keys()], vtype=GRB.INTEGER)


    def _set_constraints(self):
        if self.critical_resources:
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for j in self.K[od_pair]) == sum(1 for a in od_pair.agents if a in self.critical_resources.removed_agents) for od_pair in self.critical_resources.critical_od_pairs)
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for (od_pair, j) in self.E[v, t]) <= self.critical_resources.left_caps.get((v, t), self.G.nodes[v]["capacity"]) for (v, t) in self.E.keys())
        else:
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for j in self.K[od_pair]) == len(od_pair.agents) for od_pair in self.od_pairs)
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for (od_pair, j) in self.E[v, t]) <= self.G.nodes[v]["capacity"] for (v, t) in self.E.keys())



    def assign_solutions(self):
        if self.status == "OPTIMAL" and self.m.SolCount > 0:
            not_assigned_agents = {od_pair.id: [a for a in od_pair.agents] for od_pair in self.od_pairs} if not self.critical_resources else {od_pair.id: [a for a in od_pair.agents if a in self.critical_resources.removed_agents] for od_pair in self.critical_resources.critical_od_pairs}
            for (od_pair, j), p in self.P.items():
                n_agents = round(self.x[od_pair.id, j].X)
                if n_agents > 0:
                    delay = len(p.visits) - self.SP[od_pair.src, od_pair.dst]
                    for i in range(n_agents):
                        a = not_assigned_agents[od_pair.id][i]
                        a.path = Path(list(p.visits))
                        a.delay = delay
                    del not_assigned_agents[od_pair.id][:n_agents]
