import random
from collections import defaultdict
import multiprocessing as mp
from typing import Callable

from cfl.elements.cfl_cluster import CFL_Cluster
from cfl.elements.client import Client
from cfl.elements.facility import Facility
from general.general_environment import General_Environment, Quadrant, Coord
from utils.parallel import init_cfl, find_facilities_for_clients
from utils.read_instance import read_OR_instance, read_TBED1_instance


class CFL_Environment(General_Environment):
    def __init__(
        self,
        grid_side: int | None = None,
        max_cluster_size: int | None = None,
        n_quadrants: int | None = None,
        n_clients_per_quadrant: int | None = None,
        n_facilities_per_quadrant: int | None = None,
        offset: int | None = None,
        k: int | None = None,
        reproducibility_flag: bool | None = True,
        remove_rnd_nodes_flag: bool | None = False,
        remove_percentage: float | None = 0.1,
        seed: int | None = 42,
        instance_file: str | None = None
    ):
        super().__init__(grid_side, max_cluster_size, n_quadrants, n_clients_per_quadrant, offset, k,
                         reproducibility_flag, remove_rnd_nodes_flag, remove_percentage, seed, instance_file)

        if reproducibility_flag:
            self.rng_clients = random.Random(seed + 1_000_003)
            self.rng_facilities = random.Random(seed + 2_000_003)
        else:
            self.rng_clients = random.Random()
            self.rng_facilities = random.Random()

        self.n_facilities_per_quadrant = n_facilities_per_quadrant
        self.facilities = []
        self.quadrant_by_facility_id: dict[int, Quadrant] = {}

        self._set_environment()


    def _choose_elements(self):
        seen_positions: set[tuple[int,int]] = set()

        self._spawn_in_quadrants(
            n_per_quadrant=self.n_elements_per_quadrant,
            maker=lambda q, i: self._make_client(q, i),
            seen_key=lambda c: c.position,
            target_list=self.elements,
            quadrant_by_id=self.quadrant_by_element_id,
            start_id=0,
            seen=seen_positions,
        )

        seen_positions: set[tuple[int,int]] = set()

        self._spawn_in_quadrants(
            n_per_quadrant=self.n_facilities_per_quadrant,
            maker=lambda q, i: self._make_facility(q, i),
            seen_key=lambda f: f.position,
            target_list=self.facilities,
            quadrant_by_id=self.quadrant_by_facility_id,
            start_id=0,
            seen=seen_positions
        )

        self._start_parallel()

    def _make_client(self, quadrant: Quadrant, id_client: int) -> Client:
        pos = self._choose_position(quadrant, self.rng_clients)
        demand = 20
        return Client(id_client, pos, demand)

    def _make_facility(self, quadrant: Quadrant, id_fac: int) -> Facility:
        pos = self._choose_position(quadrant, self.rng_facilities)
        capacity = 170
        opening_cost = 1000
        return Facility(id_fac, pos, capacity, opening_cost)

    def _choose_position(self, quadrant: Quadrant, rng: random.Random) -> Coord:
        top, left = quadrant[0]
        bottom, right = quadrant[1]
        row_range = range(top, bottom + 1)
        col_range = range(left, right + 1)

        while True:
            pos = (rng.choice(row_range), rng.choice(col_range))
            if pos in self.G.nodes():
                return pos

    def _start_parallel(self):
        client_by_quadrant = defaultdict(list)
        for c in self.elements:
            q = self.quadrant_by_element_id[c.id]
            client_by_quadrant[q].append(c)

        tasks = list(client_by_quadrant.values())

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_cfl,
            initargs=(self.G, self.k, self.facilities),
        ) as pool:
            results = pool.map(find_facilities_for_clients, tasks)

        self.elements = sorted(
            (cl for group in results for cl in group),
            key=lambda cl: cl.id
        )
        fac_by_id = {f.id: f for f in self.facilities}

        for c in self.elements:
            c.k_facilities = [fac_by_id[f.id] for f in c.k_facilities if f.id in fac_by_id]
            c.all_facilities = [fac_by_id[f.id] for f in c.all_facilities if f.id in fac_by_id]


    def _build_weight(self):
        return {i: 1 for i in range(len(self.elements))}


    def _build_clusters(self, tree):
        clusters = []
        for idx, cluster_node in enumerate(tree.clusters):
            elements = sorted([self.elements[i] for i in cluster_node.cluster], key=lambda el: el.id)
            facilities = sorted(list(set(f for client in elements for f in client.k_facilities)), key=lambda f: f.id)
            clusters.append(CFL_Cluster(idx, clients=elements, facilities=facilities))

        clusters.sort(key=lambda cl: len(cl.elements), reverse=True)

        self._merge(clusters, lambda c1, c2: len(c1.elements) + len(c2.elements))
        self.clusters = sorted(clusters, key=lambda cl: cl.id)


    def solve_clusters(self, solver_factory: Callable):
        for f in self.facilities:
            f.is_open = False
            f.shipment_by_client = {}

        for c in self.elements:
            c.shipment_by_facility = {}

        self._solve_clusters(
            Client.compute_similarity,
            solver_factory,
            self._post_fn
        )


    def _post_fn(self):
        open_ids: set[int] = set()
        for hs in self.clusters_solvers:
            if hs.m.SolCount <= 0:
                continue
            for fid, var in hs.x.items():
                if var.X > 0.5:
                    open_ids.add(fid)

        for f in self.facilities:
            f.is_open = f.id in open_ids


    def _ensure_expected_occupancy_cache(self) -> None:
        if self._cluster_occ_cache is not None:
            return

        self._facility_capacity_by_id = {f.id: float(f.capacity) for f in self.facilities}

        self._cluster_occ_cache = []
        self._cluster_mass_cache = []
        occ_global = defaultdict(float)
        total_demand = 0.0

        for cluster in self.clusters:
            occ_c = defaultdict(float)
            demand_c = 0.0

            for client in cluster.elements:
                d = float(client.demand)
                demand_c += d

                K = getattr(client, "k_facilities", None) or getattr(client, "all_facilities", [])
                m = len(K)
                if m <= 0:
                    continue

                frac = d / m
                for fac in K:
                    fid = int(fac.id)
                    if fid in self._facility_capacity_by_id:
                        occ_c[fid] += frac

            self._cluster_occ_cache.append(dict(occ_c))
            self._cluster_mass_cache.append(demand_c)
            total_demand += demand_c

            for fid, val in occ_c.items():
                occ_global[fid] += val

        self._global_occ_cache = dict(occ_global)
        self._global_mass_cache = total_demand



    def _congestion_from_occ(self, occ: dict[int, float]) -> tuple[float, float]:
        cap_by_id = getattr(self, "_facility_capacity_by_id", None) or {f.id: float(f.capacity) for f in
                                                                        self.facilities}

        E_abs = 0.0
        R_max = 0.0

        for fid, occ_val in occ.items():
            cap = float(cap_by_id.get(fid, 0.0))
            if cap <= 0.0:
                E_abs += occ_val
                R_max = float("inf")
                continue

            ratio = occ_val / cap
            R_max = max(R_max, ratio)
            if occ_val > cap:
                E_abs += (occ_val - cap)

        return E_abs, R_max


    def __str__(self):
        return f"grid side: {self.grid_side}   n clients = {len(self.elements)}   n facilities = {len(self.facilities)}   k = {self.k}   n quadrants = {self.n_quadrants}"


    def _retrieve_instance(self):
        if self.instance_file.__contains__("TB") or self.instance_file.__contains__("TEST_BED_C"):
            self.G, self.elements, self.facilities = read_TBED1_instance(self.instance_file, self.k)
        else:
            self.G, self.elements, self.facilities = read_OR_instance(self.instance_file, self.k)
