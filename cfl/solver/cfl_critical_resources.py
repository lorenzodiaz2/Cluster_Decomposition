import heapq
from collections import defaultdict
from abc import ABC, abstractmethod

from cfl.elements.client import Client
from cfl.elements.facility import Facility
from general.general_critical_resources import General_Critical_Resources


class CFL_Critical_Resources(General_Critical_Resources, ABC):
    def __init__(
        self,
        clients: list[Client],
        facilities: list[Facility]
    ):
        self.facilities = facilities

        self.fac_by_id: dict[int, Facility] = {f.id: f for f in facilities}
        self.client_by_id: dict[int, Client] = {c.id: c for c in clients}

        # per sapere da quale facility sto togliendo nel passo corrente
        self._current_fid: int | None = None

        self.removed_units_by_client: dict[int, int] = defaultdict(int)

        super().__init__(clients)

        self.critical_clients: set[Client] = set()


    def _compute_residuals(self):
        return {f.id: f.capacity - sum(f.shipment_by_client.values()) for f in self.facilities}


    def _compute_relevant_resources(self):
        if not self.violated_resources:
            return None

        violated_clients_ids = set()
        for fid in self.violated_resources:
            for cid, q in self.fac_by_id[fid].shipment_by_client.items():
                if q > 0:
                    violated_clients_ids.add(cid)

        violated_clients = [self.client_by_id[cid] for cid in violated_clients_ids]

        rel = set()
        for c in violated_clients:
            rel.update(f.id for f in c.k_facilities)

        return rel


    def _get_candidates(self, resource_id: int):
        self._current_fid = resource_id
        f = self.fac_by_id[resource_id]
        return [self.client_by_id[cid] for cid, q in f.shipment_by_client.items() if q > 0]


    def _update_residuals(self):
        self.residuals = self._compute_residuals()

    def _finalize_unassign(self):
        self.left_caps = {fid: max(0, r) for fid, r in self.residuals.items()}
        self.critical_clients = {self.client_by_id[cid] for cid in self.removed_items}


    def _post_residual_update(self, fid: int, new_residual: float) -> None:
        self.residuals[fid] = new_residual

        # 5) aggiorna critical_resources + heap (lazy heap)
        if self.relevant_resources is not None and fid not in self.relevant_resources:
            self.critical_resources.discard(fid)
            return

        if new_residual < self.current_tol:
            heapq.heappush(self._heap, (new_residual, fid))
        else:
            self.critical_resources.discard(fid)

    @abstractmethod
    def _recompute(self, candidate):
        raise NotImplementedError

    @abstractmethod
    def _compute_score(self, candidate):
        raise NotImplementedError



















