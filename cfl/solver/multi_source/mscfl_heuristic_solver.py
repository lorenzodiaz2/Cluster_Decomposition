import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from cfl.elements.client import Client
from cfl.elements.facility import Facility
from cfl.solver.cfl_heuristic_solver import CFL_Heuristic_Solver
from cfl.solver.multi_source.mscfl_critical_resources import MSCFL_Critical_Resources


class MSCFL_Heuristic_solver(CFL_Heuristic_Solver):
    def __init__(
        self,
        G: nx.Graph,
        clients: list[Client],
        facilities: list[Facility],
        critical_resources: MSCFL_Critical_Resources | None = None,
        time_limit: int | None = 1800,
        verbose: bool | None = False
    ):
        super().__init__(G, clients, facilities, critical_resources, time_limit, verbose)


    def _set_variables(self):
        super()._set_variables()

        pairs = [(fid, cid) for fid, cids in self.E.items() for cid in cids]
        self.y = self.m.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0)


    def _set_constraints(self):
        if self.critical_resources:
            self.m.addConstrs(self.x[fid] == 1 for fid in self.x.keys() if self.fac_by_id[fid].is_open)
            self.m.addConstrs(gp.quicksum(self.y[f.id, c.id] for f in c.k_facilities if (f.id, c.id) in self.y) == self.repair_demand[c.id] for c in self.repair_clients)
            self.m.addConstrs(gp.quicksum(self.y[fid, cid] for cid in self.E[fid]) <= self.critical_resources.left_caps[fid] * self.x[fid] for fid in self.E.keys())
        else:
            self.m.addConstrs(gp.quicksum(self.y[f.id, c.id] for f in c.k_facilities) == c.demand for c in self.clients)
            self.m.addConstrs(gp.quicksum(self.y[fid, cid] for cid in self.E[fid]) <= self.fac_by_id[fid].capacity * self.x[fid] for fid in self.E)


    def _assign_solutions(self):
        if self.m.SolCount <= 0:
            return

        additive = bool(self.critical_resources)

        for fid in self.x.keys():
            if self.x[fid].X > 0.5:
                self.fac_by_id[fid].is_open = True
            elif not additive:
                self.fac_by_id[fid].is_open = False

        client_by_id = {c.id: c for c in self.clients}

        for (fid, cid), var in self.y.items():
            q = int(round(var.X))
            if q <= 0:
                continue

            f = self.fac_by_id[fid]
            c = client_by_id[cid]


            if additive:
                f.shipment_by_client[cid] = f.shipment_by_client.get(cid, 0) + q
                c.shipment_by_facility[fid] = c.shipment_by_facility.get(fid, 0) + q
            else:
                f.shipment_by_client[cid] = q
                c.shipment_by_facility[fid] = q


    def _compute_shipping_cost(self):
        return gp.quicksum(self.cost[(fid, cid)] * self.y[fid, cid] for (fid, cid) in self.y.keys())

