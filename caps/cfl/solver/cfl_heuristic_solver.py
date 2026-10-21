import math
from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from caps.cfl.elements.client import Client
from caps.cfl.elements.facility import Facility
from caps.cfl.solver.cfl_critical_resources import CFL_Critical_Resources
from caps.general.general_solver import General_Solver


BIG_M = 10**6

def total_solution_cost(clients, facilities, is_TB_instance: bool = True) -> int:
    open_cost = sum(fac.activation_cost for fac in facilities if fac.is_open)

    ship_cost = 0
    for cl in clients:
        for fid, q in cl.shipment_by_facility.items():
            d = cl.distances_to_facilities.get(fid, BIG_M)
            if d == math.inf:
                d = BIG_M
            ship_cost += d * q if is_TB_instance else d

    return open_cost + ship_cost



class CFL_Heuristic_Solver(General_Solver, ABC):
    def __init__(
        self,
        G: nx.Graph,
        clients: list[Client],
        facilities: list[Facility],
        critical_resources: CFL_Critical_Resources,
        time_limit: int,
        verbose: bool
    ):
        super().__init__(time_limit, verbose)
        self.G = G
        self.clients = clients
        self.facilities = facilities
        self.fac_by_id = {f.id: f for f in facilities}
        self.fixed_cost_before = 0


        self.critical_resources = critical_resources
        if critical_resources:
            if not critical_resources.is_initially_feasible:
                self.critical_resources.release_units()
            self.fixed_cost_before = total_solution_cost(self.clients, self.facilities)


        self.cost: dict[tuple[int, int], float] = {}
        self.x = None
        self.y = None
        self.E = None

        self.repair_clients: list[Client] = []
        self.repair_demand: dict[int, int] = {}

    def _set_variables(self):
        E = defaultdict(list)
        if self.critical_resources:
            self.repair_clients = [c for c in self.clients if self.critical_resources.removed_units_by_client.get(c.id, 0) > 0]
            self.repair_demand = {c.id: self.critical_resources.removed_units_by_client[c.id] for c in self.repair_clients}
            for c in self.repair_clients:
                for f in c.k_facilities:
                    if self._is_repair_candidate(c, f.id):
                        E[f.id].append(c.id)
        else:
            fac_ids = {f.id for f in self.facilities}
            for c in self.clients:
                for f in c.k_facilities:
                    if f.id in fac_ids:
                        E[f.id].append(c.id)
        self.E = dict(E)

        fac_ids = sorted(self.E.keys())
        self.x = self.m.addVars(fac_ids, vtype=GRB.BINARY)

    def _is_repair_candidate(self, client: Client, fid: int) -> bool:
        # facility utilizzabile nella repair: basta capacità residua positiva (domanda divisibile); il single source lo ridefinisce
        return self.critical_resources.left_caps.get(fid, 0) > 0

    def _set_objective(self):
        self._precompute_costs()

        if self.critical_resources:
            open_cost = gp.quicksum(self.x[fid] * self.fac_by_id[fid].activation_cost for fid in self.x.keys() if not self.fac_by_id[fid].is_open)
        else:
            open_cost = gp.quicksum(self.x[fid] * self.fac_by_id[fid].activation_cost for fid in self.x.keys())

        ship_cost = self._compute_shipping_cost()
        self.m.setObjective(open_cost + ship_cost, GRB.MINIMIZE)

    def _precompute_costs(self):
        self.cost = {}
        pairs = set(self.y.keys())
        active_clients = self.repair_clients if self.critical_resources else self.clients

        for c in active_clients:
            for f in c.k_facilities:
                key = (f.id, c.id)
                if key in pairs:
                    d = c.distances_to_facilities.get(f.id, BIG_M)
                    self.cost[key] = BIG_M if d == math.inf else d


    def _handle_infeasibility(self):
        if self.critical_resources:
            # il progresso si misura sulla quantità rilasciata: nel caso a domanda divisibile si rilasciano unità anche da clienti già rilasciati
            before = sum(self.critical_resources.removed_units_by_client.values())
            if not self.critical_resources.increment_tol():   # niente più da rilasciare: la repair si ferma (prima ciclava all'infinito)
                return False
            self.critical_resources.release_units()
            self.fixed_cost_before = total_solution_cost(self.clients, self.facilities)
            return sum(self.critical_resources.removed_units_by_client.values()) > before

        else:
            # ogni cliente del modello infeasible riceve la facility successiva in ordine di distanza (come T + 1 in MCPA);
            # le facility nuove entrano anche nel modello: per un cluster self.facilities è la lista cluster.facilities
            changed = False
            for client in self.clients:
                if len(client.k_facilities) < len(client.all_facilities):
                    client.add_facility()
                    changed = True
                    new_facility = client.k_facilities[-1]
                    if new_facility.id not in self.fac_by_id:
                        self.facilities.append(new_facility)
                        self.fac_by_id[new_facility.id] = new_facility
            return changed


    @abstractmethod
    def _set_constraints(self):
        raise NotImplementedError

    @abstractmethod
    def _assign_solutions(self):
        raise NotImplementedError

    @abstractmethod
    def _compute_shipping_cost(self):
        raise NotImplementedError