from caps.cfl.elements.client import Client
from caps.cfl.elements.facility import Facility
from caps.cfl.solver.cfl_critical_resources import CFL_Critical_Resources


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
        dist = candidate.distances_to_facilities
        n_alt = 0
        best = None  # (margin, -distance) of the best alternative facility
        for f in candidate.k_facilities:
            residual = self.residuals.get(f.id, NEG_INF)
            if f.id == fid or residual < candidate.demand:
                continue
            n_alt += 1
            key = (residual - candidate.demand, -dist[f.id])
            if best is None or key > best:
                best = key
        if best is None:
            return 0, NEG_INF, NEG_INF
        # third component: current distance minus distance of the best alternative (larger = smaller increase)
        return n_alt, best[0], dist[fid] + best[1]


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
        self.released_units.add(cid)
        self.critical_clients.add(candidate)

        # print(f"removed client {cid} with demand removed = {self.removed_units_by_client[cid]}")

        # 4) aggiorna residuo della facility
        new_residual = self.residuals[fid] + q

        # 5) parte comune
        self._post_residual_update(fid, new_residual)


