import networkx as nx
from typing import List
import gurobipy as gp
from gurobipy import GRB

from elements.pair import OD_Pair
from elements.path import Path
from solvers.general_solver import General_Solver


class CMAPF_Solver(General_Solver):
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: List[OD_Pair],
        T: int
    ):
        super().__init__(G, od_pairs)

        self.V = list(self.G.nodes())
        self.x = None
        self.y = None
        self.tau = None
        self.ub = {}


    def solve(self):
        self.set_model()
        self._optimize_model()
        while self.status != "OPTIMAL":
            self.current_T += 1
            self.set_model()
            self._optimize_model()
        self.assign_solutions()


    def set_model(self):
        N = {u: list(self.G.neighbors(u)) for u in self.V}
        A_ids = [a.id for a in self.A]

        self.m = gp.Model("exact")
        self.m.Params.OutputFlag = 0

        self.compute_ub()
        IDXs = [(a.id, v, t) for a in self.A for v in self.V for t in range(self.current_T + 1)]
        self.x = self.m.addVars(IDXs, vtype=GRB.BINARY, ub=self.ub)
        self.y = self.m.addVars(A_ids, range(self.current_T + 1), vtype=GRB.BINARY)
        self.tau = self.m.addVars(A_ids, vtype=GRB.CONTINUOUS, lb=0, ub=self.current_T)

        for a in self.A:
            s, d = a.src, a.dst

            # Inizializzazione (x e y)
            self.m.addConstrs(self.x[a.id, v, 0] == (1 if v == s else 0) for v in self.V)
            self.m.addConstr(self.y[a.id, 0] == 1)

            # Unicità di posizione
            self.m.addConstrs(gp.quicksum(self.x[a.id, v, t] for v in self.V) == self.y[a.id, t] for t in range(self.current_T + 1))

            # Transizione con attesa solo in src(a)
            for t in range(self.current_T):
                for u in self.V:
                    Ns = N[u] if u != s else N[u] + [s]
                    self.m.addConstr(self.x[a.id, u, t + 1] <= gp.quicksum(self.x[a.id, w, t] for w in Ns))
                    if u == d:
                        continue
                    self.m.addConstr(self.x[a.id, u, t] <= gp.quicksum(self.x[a.id, w, t + 1] for w in Ns))

            # No cicli / stazionarietà tranne nella sorgente e destinazione
            self.m.addConstrs(gp.quicksum(self.x[a.id, v, t] for t in range(self.current_T + 1)) <= 1 for v in self.V if v != s and v != d)

            # Niente rientro in sorgente
            self.m.addConstrs(self.x[a.id, s, t + 1] <= self.x[a.id, s, t] for t in range(self.current_T))

            # Aggiornamento della y
            self.m.addConstrs(self.y[a.id, t + 1] == self.y[a.id, t] - self.x[a.id, d, t] for t in range(self.current_T))

            # tau e ritardi
            self.m.addConstr(self.tau[a.id] == gp.quicksum(self.x[a.id, d, t] * t for t in range(self.current_T + 1)))

            # Arrivo solo una volta
            self.m.addConstr(gp.quicksum(self.x[a.id, d, t] for t in range(self.current_T + 1)) == 1)

        # Capacità dei nodi
        self.m.addConstrs(gp.quicksum(self.x[a, v, t] for a in A_ids) <= self.G.nodes[v]["capacity"] for v in self.V for t in range(self.current_T + 1))

        # Obiettivo
        self.m.setObjective(gp.quicksum(self.tau[a.id] - self.SP[a.src, a.dst] for a in self.A), GRB.MINIMIZE)


    def assign_solutions(self):
        if self.status == "OPTIMAL" and self.m.SolCount > 0:
            N = {u: list(self.G.neighbors(u)) for u in self.V}

            for od_pair in self.od_pairs:
                SP = self.SP[od_pair.src, od_pair.dst]
                for a in od_pair.agents:
                    a.path = Path([a.src])
                    for t in range(1, int(self.tau[a.id].X) + 1):
                        current = a.path.visits[t - 1]
                        outs = N[current] + ([current] if current == a.src else [])
                        for v in outs:
                            if self.x[a.id, v, t].X > 0.5:
                                current = v
                                break
                        a.path.visits.append(current)
                    a.delay = int(self.tau[a.id].X) - SP



    def compute_ub(self):
        src_to_node, dst_to_node = self.build_distances()
        self.ub = {}
        for a in self.A:
            s, d = a.src, a.dst
            sp = self.SP[s, d]
            for v in self.V:
                ds = src_to_node[s, v]
                dd = dst_to_node[d, v]
                for t in range(self.current_T + 1):
                    ban = (
                        ds == float('inf') or dd == float('inf') or
                        ds > t or dd > (self.current_T - t)
                    )
                    # corridor (opzionale): setta, e.g., self.W = 2 o 3
                    if hasattr(self, "W"):
                        ban = ban or (ds + dd > sp + self.W)
                    if ban:
                        self.ub[a.id, v, t] = 0.0


    def build_distances(self):
        srcs = {od.src for od in self.od_pairs}
        dsts = {od.dst for od in self.od_pairs}
        nodes = srcs | dsts
        INF = float('inf')

        # dizionario di dizionari
        dist = {s: nx.single_source_shortest_path_length(self.G, s) for s in nodes}

        src_to_node = {(s, v): dist[s].get(v, INF) for s in srcs for v in self.V}
        dst_to_node = {(d, v): dist[d].get(v, INF) for d in dsts for v in self.V}

        return src_to_node, dst_to_node
