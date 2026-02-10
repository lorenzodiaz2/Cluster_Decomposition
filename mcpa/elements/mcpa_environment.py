from collections import defaultdict
import multiprocessing as mp

from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from general.general_environment import General_Environment, Quadrant
from mcpa.elements.agent import Agent
from mcpa.elements.mcpa_cluster import MCPA_Cluster
from mcpa.elements.pair import OD_Pair
from mcpa.elements.path import Path
from utils.parallel import init_mcpa, compute_paths_for_quadrant


def _build_occ_for_ods(ods) -> dict[tuple[int, int], float]:
    occ: dict[tuple[int, int], float] = {}

    for od in ods:
        n_agents_od = len(od.agents)
        paths = od.all_paths
        n_paths = len(paths)

        visit_counts: dict[tuple[int, int], int] = defaultdict(int)
        for path in paths:
            enc = path.encoded
            for t, node_id in enumerate(enc):
                visit_counts[(t, int(node_id))] += 1

        for key, count_paths in visit_counts.items():
            frac = count_paths / n_paths
            occ[key] = occ.get(key, 0.0) + n_agents_od * frac

    return occ



class MCPA_Environment(General_Environment):
    def __init__(
        self,
        grid_side: int,
        max_cluster_size: int,
        n_quadrants: int,
        n_pairs_per_quadrant: int,
        offset: int,
        k: int,
        reproducibility_flag: bool | None = True,
        remove_rnd_nodes_flag: bool | None = False,
        remove_percentage: float | None = 0.1,
        seed: int | None = 42,
        instance_file: str | None = None
    ):
        super().__init__(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, offset, k,
                         reproducibility_flag, remove_rnd_nodes_flag, remove_percentage, seed, instance_file)

        Path.set_grid_side(grid_side)
        self.agents: list[Agent] = []

        self._set_environment()


    def _start_parallel(self):
        od_by_quadrant: dict[Quadrant, list[OD_Pair]] = defaultdict(list)
        for od in self.elements:
            q = self.quadrant_by_element_id[od.id]
            od_by_quadrant[q].append(od)

        tasks = list(od_by_quadrant.values())

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_mcpa,
            initargs=(self.G, self.k),
        ) as pool:
            results = pool.map(compute_paths_for_quadrant, tasks)

        self.elements = sorted(
            (pair for group in results for pair in group),
            key=lambda pair: pair.id
        )
        self.agents = [a for od in self.elements for a in od.agents]


    def _build_weight(self):
        return {i: len(self.elements[i].agents) for i in range(len(self.elements))}

    def _build_clusters(self, tree):
        clusters = [
            MCPA_Cluster(idx, [self.elements[i] for i in cluster_node.cluster])
            for idx, cluster_node in enumerate(tree.clusters)
        ]

        self._merge(clusters, lambda c1, c2: c1.n_agents + c2.n_agents)
        self.clusters = clusters


    def _choose_elements(self):
        self._next_agent_id = 0

        self._spawn_in_quadrants(
            n_per_quadrant=self.n_elements_per_quadrant,
            maker=lambda q, i: self._make_od_pair(q, i),
            seen_key=lambda od: (od.src, od.dst),
            target_list=self.elements,
            quadrant_by_id=self.quadrant_by_element_id,
            start_id=0,
            seen=set(),
        )

        self._set_capacities()
        self._start_parallel()

    def _make_od_pair(self, quadrant: Quadrant, id_pair: int) -> OD_Pair:
        top, left = quadrant[0]
        bottom, right = quadrant[1]
        row_range = range(top, bottom + 1)
        col_range = range(left, right + 1)

        while True:
            src = (self.rng.choice(row_range), self.rng.choice(col_range))
            dst = (self.rng.choice(row_range), self.rng.choice(col_range))
            if src == dst:
                continue
            if src not in self.G.nodes() or dst not in self.G.nodes():
                continue
            break

        n_agents = self.rng.randrange(5, 6)
        start = self._next_agent_id
        agents = [Agent(i, src, dst) for i in range(start, start + n_agents)]
        self._next_agent_id += n_agents

        return OD_Pair(id_pair, src, dst, agents)


    def _set_capacities(self):
        od_nodes = [node for od_pair in self.elements for node in (od_pair.src, od_pair.dst)]
        V = list(self.G.nodes)
        for v in V:
            self.G.nodes[v]['capacity'] = 0

        for od_pair in self.elements:
            k = len(od_pair.agents)
            self.G.nodes[od_pair.src]['capacity'] += k
            self.G.nodes[od_pair.dst]['capacity'] += k

        for v in V:
            if v not in od_nodes:
                self.G.nodes[v]["capacity"] = self.rng.randrange(5, 6)  #############


    def solve_clusters(self):
        self._solve_clusters(
            OD_Pair.compute_similarity,
            lambda env, c: MCPA_Heuristic_Solver(env.G, c.elements)
        )



    def _congestion_from_occ(self, occ: dict[tuple[int, int], float]) -> tuple[float, float]:
        n_side = self.grid_side
        E_abs = 0.0
        R_max = 0.0

        for (t, node_id), occ_val in occ.items():
            i = node_id // n_side
            j = node_id % n_side
            v = (i, j)

            cap = float(self.G.nodes[v]["capacity"])
            ratio = occ_val / cap

            if ratio > R_max:
                R_max = ratio

            if occ_val > cap:
                E_abs += (occ_val - cap)

        return E_abs, R_max



    def _ensure_expected_occupancy_cache(self) -> None:
        if self._cluster_occ_cache is not None:
            return

        self._cluster_occ_cache = []
        occ_global = defaultdict(float)

        self._cluster_mass_cache = [float(cluster.n_agents) for cluster in self.clusters]
        self._global_mass_cache = float(len(self.agents))

        for cluster in self.clusters:
            occ_c = _build_occ_for_ods(cluster.elements)
            self._cluster_occ_cache.append(occ_c)
            for k, v in occ_c.items():
                occ_global[k] += v

        self._global_occ_cache = dict(occ_global)


    def _retrieve_instance(self):
        """Se voglio recuperare un'istanza da un dataset tocca implementa sto metodo"""
        pass

    def __str__(self):
        return f"grid side: {self.grid_side}   n OD = {len(self.elements)}   k = {self.k}   n quadrants = {self.n_quadrants}"


