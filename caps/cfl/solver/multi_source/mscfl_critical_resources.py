import math
from caps.cfl.elements.client import Client
from caps.cfl.elements.facility import Facility
from caps.cfl.solver.cfl_critical_resources import CFL_Critical_Resources

EPS = 1e-6


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

        prev = f.shipment_by_client.get(cid, 0)
        if prev <= EPS:
            f.shipment_by_client.pop(cid, None)
            candidate.shipment_by_facility.pop(fid, None)
            return

        # 1) fractional-first: release the decimal part, otherwise one unit (never more than prev)
        frac = prev - math.floor(prev)
        if EPS < frac < 1 - EPS:
            delta = frac
        else:
            delta = min(1, prev)

        # 2) update both sides consistently
        new = prev - delta
        if new <= EPS:
            f.shipment_by_client.pop(cid, None)
            candidate.shipment_by_facility.pop(fid, None)
        else:
            f.shipment_by_client[cid] = new
            candidate.shipment_by_facility[fid] = new

        # 3) bookkeeping
        self.removed_units_by_client[cid] += delta
        self.removed_items.add(cid)
        self.critical_clients.add(candidate)

        # 4) update the residual by the amount actually released
        self._post_residual_update(fid, self.residuals[fid] + delta)





