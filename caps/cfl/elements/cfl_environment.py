import math
import random
from collections import defaultdict
import multiprocessing as mp
from typing import Callable

from caps.cfl.elements.cfl_cluster import CFL_Cluster
from caps.cfl.elements.client import Client
from caps.cfl.elements.facility import Facility
from caps.general.general_environment import General_Environment, Quadrant, Coord
from caps.cfl.elements.cfl_parallel import init_cfl, find_facilities_for_clients
from utils.instance_generators import make_hotspots, sample_point_hotspot
from utils.read_instance import read_or_instance, read_tbed1_instance


class CFL_Environment(General_Environment):
    def __init__(
        self,
        grid_side: int | None = None,
        max_cluster_size: int | None = None,
        n_quadrants: int | None = None,
        n_clients_per_quadrant: int | None = None,
        n_facilities_per_quadrant: int | None = None,
        resources_capacities: int | None = None,
        offset: int | None = None,
        k: int | None = None,
        reproducibility_flag: bool | None = True,
        remove_rnd_nodes_flag: bool | None = False,
        remove_percentage: float | None = 0.1,
        seed: int | None = 42,
        sim_method: tuple[str, float | None] | None = ("uniform", None),
        resource_weights: str = "linear",
        instance_file: str | None = None,
        partitioner: str = "nj",
        family: str = "quadrants",
        family_params: dict | None = None,
        generator: str = "legacy",
        capacity_ratio: float = 3.0,
        opening_cost_scale: float = 1.0,
        demand_range: tuple[int, int] = (5, 35),
        capacity_range: tuple[float, float] = (10, 160)
    ):
        super().__init__(grid_side, max_cluster_size, n_quadrants, n_clients_per_quadrant, resources_capacities, offset, k,
                         reproducibility_flag, remove_rnd_nodes_flag, remove_percentage, seed, sim_method, resource_weights, instance_file,
                         partitioner=partitioner, family=family, family_params=family_params)

        self.rng_clients = random.Random(seed if reproducibility_flag else None)
        self.rng_facilities = random.Random((seed + 1) if reproducibility_flag else None)

        self.n_facilities_per_quadrant = n_facilities_per_quadrant
        self.facilities = []
        self.quadrant_by_facility_id: dict[int, Quadrant] = {}

        # "legacy": domanda 20, capacità resources_capacities e costo di apertura 1000 uguali per tutti (istanze della tesi)
        # "cornuejols": generatore di Cornuéjols, Sridharan e Thizy (1991) -> domande, capacità e costi eterogenei
        if generator not in ("legacy", "cornuejols"):
            raise ValueError(f"generator '{generator}' non supportato")
        self.generator = generator
        self.capacity_ratio = capacity_ratio            # r = capacità totale / domanda totale, imposto in ogni quadrante
        self.opening_cost_scale = opening_cost_scale    # moltiplicatore dei costi di apertura (da tarare)
        self.demand_range = demand_range
        self.capacity_range = capacity_range
        self._opening_cost_draws: dict[int, tuple[float, float]] = {}

        self._set_environment()


    def _choose_elements(self):
        # famiglie: "quadrants" (come prima), "uniform" (clienti e facility uniformi su tutta la griglia),
        # "hotspots" (clienti attorno a hotspot, facility uniformi); per le ultime due serve un solo quadrante = tutta la griglia
        if self.family not in ("quadrants", "uniform", "hotspots"):
            raise ValueError(f"family '{self.family}' non supportata")
        if self.family != "quadrants" and self.n_quadrants != 1:
            raise ValueError(f"la famiglia '{self.family}' richiede n_quadrants = 1")
        if self.family == "hotspots":
            self._nodes = list(self.G.nodes)
            self._node_set = set(self._nodes)
            self._hotspot_centers, self._hotspot_weights = make_hotspots(self._nodes, self.rng_clients, self.family_params["n_hotspots"])

        seen_positions: set[tuple[int, int]] = set()
        if self.n_elements_per_quadrant:
            self._spawn_in_quadrants(
                n_per_quadrant=self.n_elements_per_quadrant,
                maker=lambda q, i: self._make_client(q, i),
                seen_key=lambda c: c.position,
                target_list=self.elements,
                quadrant_by_id=self.quadrant_by_element_id,
                start_id=0,
                seen=seen_positions,
            )

        seen_positions: set[tuple[int, int]] = set()
        if self.n_facilities_per_quadrant:
            self._spawn_in_quadrants(
                n_per_quadrant=self.n_facilities_per_quadrant,
                maker=lambda q, i: self._make_facility(q, i, self.resources_capacity),
                seen_key=lambda f: f.position,
                target_list=self.facilities,
                quadrant_by_id=self.quadrant_by_facility_id,
                start_id=0,
                seen=seen_positions
            )

        if self.generator == "cornuejols":
            self._apply_cornuejols()   # prima della ricerca delle facility più vicine, che usa le capacità

        self._start_parallel()
        weighting, beta = self.sim_method if self.sim_method else ("uniform", None)
        for client in self.elements:
            client.build_option_weights(weighting, beta)


    def _make_client(self, quadrant: Quadrant, id_client: int) -> Client:
        if self.family == "hotspots":
            pos = sample_point_hotspot(self._nodes, self._node_set, self.rng_clients, self._hotspot_centers, self._hotspot_weights,
                                       self.family_params["sigma"], self.family_params.get("noise", 0.0))
        else:
            pos = self._choose_position(quadrant, self.rng_clients)   # con un solo quadrante: uniforme su tutta la griglia
        if self.generator == "cornuejols":
            demand = self.rng_clients.randint(*self.demand_range)
        else:
            demand = 20
        return Client(id_client, pos, demand)


    def _make_facility(self, quadrant: Quadrant, id_fac: int, cap: int | None = 0) -> Facility:
        pos = self._choose_position(quadrant, self.rng_facilities)
        if self.generator == "cornuejols":
            # capacità grezza (riscalata dopo in _apply_cornuejols) e le due estrazioni del costo di apertura
            capacity = self.rng_facilities.uniform(*self.capacity_range)
            self._opening_cost_draws[id_fac] = (self.rng_facilities.uniform(0, 90), self.rng_facilities.uniform(100, 110))
            opening_cost = 0
        else:
            capacity = cap
            opening_cost = 1000
        return Facility(id_fac, pos, capacity, opening_cost)


    def _apply_cornuejols(self):
        """
        Riscala le capacità in modo che in ogni quadrante capacità totale / domanda totale = capacity_ratio (capacità intere, almeno 1),
        poi calcola i costi di apertura f_j = (U[0, 90] + U[100, 110] * sqrt(s_j)) * opening_cost_scale, arrotondati a intero.
        """
        demand_by_quadrant = defaultdict(float)
        for c in self.elements:
            demand_by_quadrant[self.quadrant_by_element_id[c.id]] += c.demand

        facilities_by_quadrant = defaultdict(list)
        for f in self.facilities:
            facilities_by_quadrant[self.quadrant_by_facility_id[f.id]].append(f)

        for q, facs in facilities_by_quadrant.items():
            raw_total = sum(f.capacity for f in facs)
            target = self.capacity_ratio * demand_by_quadrant.get(q, 0.0)
            factor = target / raw_total if raw_total > 0 else 0.0
            for f in facs:
                f.capacity = max(1, round(f.capacity * factor))
                a, b = self._opening_cost_draws[f.id]
                f.activation_cost = round((a + b * math.sqrt(f.capacity)) * self.opening_cost_scale)

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
        # con pochi quadranti (es. famiglie senza quadranti) divido comunque il lavoro tra i processi
        if len(tasks) < mp.cpu_count():
            all_clients = [c for task in tasks for c in task]
            n_chunks = min(mp.cpu_count(), len(all_clients))
            tasks = [all_clients[i::n_chunks] for i in range(n_chunks)]

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
        return {i: 1 for i in range(len(self.elements))}   # -> ogni cliente ha peso 1 per l'albero filogenetico


    def _build_clusters(self, tree):
        clusters = []
        for idx, cluster_node in enumerate(tree.clusters):
            elements = sorted([self.elements[i] for i in cluster_node.cluster], key=lambda el: el.id)
            facilities = sorted(list(set(f for client in elements for f in client.k_facilities)), key=lambda f: f.id)
            clusters.append(CFL_Cluster(idx, clients=elements, facilities=facilities))

        clusters.sort(key=lambda cl: len(cl.elements), reverse=True)

        self._merge(clusters, lambda c1, c2: len(c1.elements) + len(c2.elements))
        self.clusters = sorted(clusters, key=lambda cl: cl.id)

    def _compute_resource_weights(self) -> dict[int, float]:
        weights = {}
        D = defaultdict(float)

        for client in self.elements:
            b_e = float(client.demand)
            for f_id, pi_er in client.option_weights.items():
                D[f_id] += b_e * pi_er

        for facility in self.facilities:
            cap = float(facility.capacity)
            d_val = D.get(facility.id, 0.0)

            # Se la facility ha capacità 0, previene division by zero
            ratio = d_val / cap if cap > 0 else 0.0

            if self.resource_weights == "linear":
                w = ratio
            elif self.resource_weights == "exponential":
                w = math.exp(ratio - 1.0)
            else:
                w = 1.0
            weights[facility.id] = w
        return weights


    def solve_clusters(self, solver_factory: Callable):
        for f in self.facilities:
            f.is_open = False
            f.shipment_by_client = {}

        for c in self.elements:
            c.shipment_by_facility = {}

        resource_weights = self._compute_resource_weights() if self.resource_weights != "uniform" else {}

        self._solve_clusters(
            Client.compute_similarity,
            solver_factory,
            self._post_fn,
            resource_weights=resource_weights
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

                for fid, pi_er in client.option_weights.items():
                    if fid in self._facility_capacity_by_id:
                        occ_c[fid] += d * pi_er

            self._cluster_occ_cache.append(dict(occ_c))
            self._cluster_mass_cache.append(demand_c)
            total_demand += demand_c

            for fid, val in occ_c.items():
                occ_global[fid] += val

        self._global_occ_cache = dict(occ_global)
        self._global_mass_cache = total_demand



    def _congestion_from_occ(self, occ: dict[int, float]) -> tuple[float, float]:
        cap_by_id = getattr(self, "_facility_capacity_by_id", None) or {f.id: float(f.capacity) for f in self.facilities}

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

    def _get_active_resource_capacities(self) -> dict[int, float]:
        cap_by_id = getattr(self, "_facility_capacity_by_id", None) or {f.id: float(f.capacity) for f in
                                                                        self.facilities}
        return {fid: cap_by_id.get(fid, 0.0) for fid in self._global_occ_cache.keys()}

    def _compute_flexibility_metrics(self, occ_global: dict[int, float], cap_dict: dict[int, float]) -> tuple[
        float, float, float]:
        n_active_res = len(occ_global)
        if n_active_res == 0 or not self.elements:
            return 0.0, 0.0, 0.0

        n_elements = len(self.elements)
        total_options_breadth = 0
        guaranteed_safe_entities = 0

        for client in self.elements:
            # Breadth: numero di facility k_nearest esplorate
            total_options_breadth += len(client.k_facilities)

            # Slack: Esiste almeno una facility nel pool di questo client non satura?
            has_safe_facility = False
            for f in client.k_facilities:
                if occ_global.get(f.id, 0.0) <= cap_dict.get(f.id, float('inf')):
                    has_safe_facility = True
                    break

            if has_safe_facility:
                guaranteed_safe_entities += 1

        b_opt = (total_options_breadth / n_elements) / n_active_res
        f_slack = guaranteed_safe_entities / n_elements

        total_space = sum(cap_dict.values())
        total_footprint_volume = sum(occ_global.values())
        r_sm = float(total_space / total_footprint_volume) if total_footprint_volume > 0 else 0.0

        return b_opt, f_slack, r_sm


    def _retrieve_instance(self):
        if self.instance_file:
            if self.k and ("TB" in self.instance_file or "TEST_BED_C" in self.instance_file):
                self.G, self.elements, self.facilities = read_tbed1_instance(self.instance_file, self.k)
            else:
                self.G, self.elements, self.facilities = read_or_instance(self.instance_file, self.k)

        weighting, beta = self.sim_method if self.sim_method else ("uniform", None)
        for client in self.elements:
            client.build_option_weights(weighting, beta)