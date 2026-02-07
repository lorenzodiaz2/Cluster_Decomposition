from cfl.elements.client import Client
from cfl.elements.facility import Facility
from cfl.solver.cfl_critical_resources import CFL_Critical_Resources


class MSCFL_Critical_Resources(CFL_Critical_Resources):
    def __init__(
        self,
        clients: list[Client],
        facilities: list[Facility]
    ):
        super().__init__(clients, facilities)


    def _compute_score(self, candidate: Client):
        return sum(max(0, self.residuals[f.id]) for f in candidate.k_facilities)

    def _recompute(self, candidate: Client):
        fid = self._current_fid
        assert fid is not None

        f = self.fac_by_id[fid]
        cid = candidate.id

        # 1) togli 1 unità dalla facility -> client
        prev = f.shipment_by_client[cid]
        if prev == 1:
            f.shipment_by_client.pop(cid)
        else:
            f.shipment_by_client[cid] -= 1

        # 2) mantieni coerenza anche lato client
        prev_c = candidate.shipment_by_facility[fid]
        if prev_c <= 1:
            candidate.shipment_by_facility.pop(fid)
        else:
            candidate.shipment_by_facility[fid] -= 1

        # 3) bookkeeping
        self.removed_units_by_client[cid] += 1
        self.removed_items.add(cid)
        self.critical_clients.add(candidate)

        # 4) aggiorna residuo della facility
        new_residual = self.residuals[fid] + 1

        # 5) parte comune
        self._post_residual_update(fid, new_residual)



