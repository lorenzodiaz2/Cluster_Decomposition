from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from mcpa.elements.agent import Agent
from mcpa.elements.pair import OD_Pair
from mcpa.elements.path import Path
from general.general_solver import General_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources


class MCPA_Heuristic_Solver(General_Solver):
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: list[OD_Pair],
        critical_resources: MCPA_Critical_Resources | None = None,
        time_limit: int | None = 1800,
        verbose : bool | None = False
    ):
        super().__init__(time_limit, verbose)

        self.P = None
        self.K = None
        self.E = None
        self.x = None
        self.critical_resources = critical_resources
        if critical_resources:
            if not critical_resources.is_initially_feasible:
                self.critical_resources.unassign_items()
        self.G = G
        self.od_pairs = od_pairs
        self.A: list[Agent] = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.SP = {(od.src, od.dst): len(od.k_shortest_paths[0].visits) for od in self.od_pairs} ############# lunghezza totale, NON in passi



    def _handle_infeasibility(self):
        if self.critical_resources:
            before = len(self.critical_resources.removed_items)
            self.critical_resources.increment_tol()
            self.critical_resources.unassign_items()
            after = len(self.critical_resources.removed_items)
        else:
            before = sum(len(od.all_paths) for od in self.od_pairs)
            for od_pair in self.od_pairs:
                od_pair.delay_shortest_paths(od_pair.T + 1)
                od_pair.T += 1
            after = sum(len(od.all_paths) for od in self.od_pairs)
        return after > before


    def _set_objective(self):
        self.m.setObjective(gp.quicksum(self.x[od_pair.id, j] * (len(p.visits) - self.SP[od_pair.src, od_pair.dst]) for (od_pair, j), p in self.P.items()), GRB.MINIMIZE)


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
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for j in self.K[od_pair]) == sum(1 for a in od_pair.agents if a in self.critical_resources.removed_items) for od_pair in self.critical_resources.critical_od_pairs)
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for (od_pair, j) in self.E[v, t]) <= self.critical_resources.left_caps.get((v, t), self.G.nodes[v]["capacity"]) for (v, t) in self.E.keys())
        else:
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for j in self.K[od_pair]) == len(od_pair.agents) for od_pair in self.od_pairs)
            self.m.addConstrs(gp.quicksum(self.x[od_pair.id, j] for (od_pair, j) in self.E[v, t]) <= self.G.nodes[v]["capacity"] for (v, t) in self.E.keys())



    def _assign_solutions(self):
        if self.status == "OPTIMAL" and self.m.SolCount > 0:
            not_assigned_agents = {od_pair.id: [a for a in od_pair.agents] for od_pair in self.od_pairs} if not self.critical_resources else {od_pair.id: [a for a in od_pair.agents if a in self.critical_resources.removed_items] for od_pair in self.critical_resources.critical_od_pairs}
            for (od_pair, j), p in self.P.items():
                n_agents = round(self.x[od_pair.id, j].X)
                if n_agents > 0:
                    delay = len(p.visits) - self.SP[od_pair.src, od_pair.dst]
                    for i in range(n_agents):
                        a = not_assigned_agents[od_pair.id][i]
                        a.path = Path(list(p.visits))
                        a.delay = delay
                    del not_assigned_agents[od_pair.id][:n_agents]
