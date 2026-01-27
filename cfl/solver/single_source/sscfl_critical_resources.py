from cfl.elements.client import Client
from cfl.elements.facility import Facility
from cfl.solver.cfl_critical_resources import CFL_Critical_Resources


class SSCFL_Critical_Resources(CFL_Critical_Resources):
    def __init__(
        self,
        clients: list[Client],
        facilities: list[Facility]
    ):
        super().__init__(clients, facilities)


    def _compute_score(self, candidate: Client):
        NEG_INF = float("-inf")
        fid = self._current_fid
        margins = [
            self.residuals.get(f.id, NEG_INF) - candidate.demand
            for f in candidate.k_facilities
            if f.id != fid and self.residuals.get(f.id, NEG_INF) >= candidate.demand
        ]
        return len(margins), max(margins, default=NEG_INF)


    def _recompute(self, candidate: Client):
        fid = self._current_fid
        assert fid is not None

        f = self.fac_by_id[fid]
        cid = candidate.id

        q = f.shipment_by_client.get(cid, 0)
        if q <= 0:
            self.critical_resources.discard(fid)
            return

        # 1) togli TUTTO dalla facility -> client
        f.shipment_by_client.pop(cid, None)

        # 2) coerenza lato client
        candidate.shipment_by_facility.pop(fid, None)

        # 3) bookkeeping
        prev_removed = self.removed_units_by_client.get(cid, 0)
        self.removed_units_by_client[cid] = max(prev_removed, int(q))
        self.removed_items.add(cid)
        self.critical_clients.add(candidate)

        print(f"removed client {cid} with demand removed = {self.removed_units_by_client[cid]}")

        # 4) aggiorna residuo della facility
        new_residual = self.residuals[fid] + q

        # 5) parte comune
        self._post_residual_update(fid, new_residual)


