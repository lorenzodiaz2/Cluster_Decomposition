import random
import time
from collections import defaultdict
from typing import List
import multiprocessing as mp
import networkx as nx
import numpy as np

from elements.agent import Agent
from elements.cluster import Cluster
from elements.pair import OD_Pair
from elements.path import Path
from nj.tree_partition import TreePartition
from utils.grid_utils import set_quadrants_4_9
from utils.parallel import init_paths_pool, compute_paths_for_quadrant, init_similarity_pool, compute_similarity_row

Coord = tuple[int, int]
Quadrant = tuple[Coord, Coord]


class Environment:
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
        restrict_paths_to_quadrant: bool | None = False
    ):
        self.grid_side = grid_side
        Path.set_grid_side(grid_side)
        self.max_cluster_size = max_cluster_size
        self.n_quadrants = n_quadrants
        self.n_pairs_per_quadrant = n_pairs_per_quadrant
        self.offset = offset
        self.k = k
        self.G = None
        self.od_pairs: List[OD_Pair] = []
        self.agents: List[Agent] = []
        self.clusters: List[Cluster] = []
        self.similarity_matrix = None
        self.set_time = None
        self.matrix_time = None
        self.nj_time = None
        self.rng = random.Random(seed if reproducibility_flag else None)
        self.remove_rnd_nodes_flag = remove_rnd_nodes_flag
        self.remove_percentage = remove_percentage
        self.quadrants: List[Quadrant] = []
        self.quadrant_by_od: dict[int, Quadrant] = {}
        self.restrict_paths_to_quadrant = restrict_paths_to_quadrant

        self.similarity_index = None
        self.cluster_similarity_indexes = None
        self.cluster_congestion_indexes = None
        self.cluster_congestion_indexes_absolute = None
        self.cluster_congestion_ratio_max = None

        self._od_index: dict[OD_Pair, int] | None = None

        self._set_environment()


    def _set_environment(self):
        start = time.perf_counter()
        self._create_grid_graph()
        self._choose_pairs()
        self._set_capacities()

        od_by_quadrant: dict[Quadrant, list[OD_Pair]] = defaultdict(list)
        for od in self.od_pairs:
            q = self.quadrant_by_od[od.id]
            od_by_quadrant[q].append(od)

        tasks = list(od_by_quadrant.items())

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_paths_pool,
            initargs=(self.G, self.k, self.restrict_paths_to_quadrant),
        ) as pool:
            results = pool.map(compute_paths_for_quadrant, tasks)

        self.od_pairs = [od for group in results for od in group]


        self.agents = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.set_time = time.perf_counter() - start


    def compute_clusters(
        self,
        refinement_levels: int = 2,
        E_abs_threshold: float = 250.0,
        R_max_threshold: float = 2.0,
    ):
        start = time.perf_counter()
        n = len(self.od_pairs)
        self.similarity_matrix = np.zeros((n, n), dtype=int)

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_similarity_pool,
            initargs=(self.od_pairs,),
        ) as pool:
            results = pool.map(compute_similarity_row, range(n - 1))

        for i, row in results:
            self.similarity_matrix[i, i + 1:] = row[i + 1:]
            self.similarity_matrix[i + 1:, i] = row[i + 1:]

        self.matrix_time = time.perf_counter() - start

        start = time.perf_counter()
        tree = TreePartition(self.similarity_matrix, self.od_pairs, self.max_cluster_size)
        self.clusters = tree.compute_clusters()


        self._od_index = {od: idx for idx, od in enumerate(self.od_pairs)}

        self._compute_cluster_congestion_indexes()

        base_max_size = self.max_cluster_size

        for refinement_level in range(1, refinement_levels + 1):
            level_max_size = max(1, base_max_size // (2 ** refinement_level)) + 1

            E_abs_list = self.cluster_congestion_indexes_absolute
            R_max_list = self.cluster_congestion_ratio_max

            new_clusters: list[Cluster] = []
            any_split = False

            for C, E_abs, R_max in zip(self.clusters, E_abs_list, R_max_list):
                should_split = (
                    (E_abs is not None and E_abs > E_abs_threshold) or
                    (R_max is not None and R_max > R_max_threshold)
                )

                if should_split and C.n_agents > level_max_size and len(C.od_pairs) > 1:
                    new_clusters.extend(self._split_cluster(C, level_max_size))
                    any_split = True
                else:
                    new_clusters.append(C)

            if not any_split:
                break

            self.clusters = new_clusters
            self._compute_cluster_congestion_indexes()

        self.nj_time = time.perf_counter() - start
        self._compute_similarity_index()


    def _split_cluster(
        self,
        cluster: Cluster,
        max_cluster_size: int
    ) -> list[Cluster]:
        if len(cluster.od_pairs) <= 1:
            return [cluster]

        idxs = [self._od_index[od] for od in cluster.od_pairs]
        sub = self.similarity_matrix[np.ix_(idxs, idxs)]

        tree = TreePartition(sub, list(cluster.od_pairs), max_cluster_size)
        new_clusters = tree.compute_clusters()

        last_id = max(c.id for c in self.clusters) if self.clusters else -1
        for k, c in enumerate(new_clusters, start=1):
            c.id = last_id + k

        return new_clusters


    def _create_grid_graph(self):
        self.G = nx.Graph()

        n = self.grid_side
        self.G.add_nodes_from((i, j) for i in range(n) for j in range(n))
        self.G.add_edges_from([((i, j), (i + 1, j)) for i in range(n - 1) for j in range(n)] +
                         [((i, j), (i, j + 1)) for i in range(n) for j in range(n - 1)])


        if self.remove_rnd_nodes_flag:
            V = self.G.nodes()
            self.G.remove_nodes_from({v for v in V if self.rng.random() < self.remove_percentage})
            self.G.remove_nodes_from({v for v in V if self.G.degree(v) == 0})


    def _choose_pairs(self):
        if self.n_quadrants == 1:
            self.quadrants = [((0, 0), (self.grid_side - 1, self.grid_side - 1))]
        else:
            self.quadrants = set_quadrants_4_9(self.grid_side, self.n_quadrants, self.offset)
        i = 0
        n_tot_agents = 0
        seen = set()
        # for quadrant in self.quadrants:
        #     for _ in range(self.n_pairs_per_quadrant):
        #         od_pair = self._choose_pair(quadrant, i, n_tot_agents, seen)
        #         self.od_pairs.append(od_pair)
        #         i += 1
        #         n_tot_agents += len(od_pair.agents)
        #         seen.add((od_pair.src, od_pair.dst))
        #         self.quadrant_by_od[od_pair.id] = quadrant

        for j, quadrant in enumerate(self.quadrants):
            if j < self.n_quadrants:
                for _ in range(self.n_pairs_per_quadrant):
                    od_pair = self._choose_pair(quadrant, i, n_tot_agents, seen)
                    self.od_pairs.append(od_pair)
                    i += 1
                    n_tot_agents += len(od_pair.agents)
                    seen.add((od_pair.src, od_pair.dst))
                    self.quadrant_by_od[od_pair.id] = quadrant


    def _choose_pair(
        self,
        quadrant: Quadrant,
        id_pair: int,
        n_tot_agents: int,
        seen
    ) -> OD_Pair:
        top, left = quadrant[0]
        bottom, right = quadrant[1]

        row_range = range(top, bottom + 1)
        col_range = range(left, right + 1)

        while True:
            src = (self.rng.choice(row_range), self.rng.choice(col_range))
            dst = (self.rng.choice(row_range), self.rng.choice(col_range))
            if src == dst or src not in self.G.nodes() or dst not in self.G.nodes() or (src, dst) in seen:
                continue
            break

        number_of_agents = self.rng.randrange(5, 6) #############
        agents = [Agent(i, src, dst) for i in range(n_tot_agents, n_tot_agents + number_of_agents)]

        return OD_Pair(id_pair, src, dst, agents)



    def _set_capacities(self):
        od_nodes = [node for od_pair in self.od_pairs for node in (od_pair.src, od_pair.dst)]
        V = list(self.G.nodes)
        for v in V:
            self.G.nodes[v]['capacity'] = 0

        for od_pair in self.od_pairs:
            k = len(od_pair.agents)
            self.G.nodes[od_pair.src]['capacity'] += k
            self.G.nodes[od_pair.dst]['capacity'] += k

        for v in V:
            if v not in od_nodes:
                self.G.nodes[v]["capacity"] = self.rng.randrange(5, 6)  #############


    def _compute_similarity_index(self):
        S = self.similarity_matrix
        sim_total = S.sum() / 2

        if sim_total == 0:
            self.similarity_index = 0.0
            self.cluster_similarity_indexes = [0.0 for _ in self.clusters]
            return

        sim_intra_global = 0.0
        cluster_indexes = []

        n = S.shape[0]
        all_ids = np.arange(n)

        for cluster in self.clusters:
            od_ids = np.array([self._od_index[od] for od in cluster.od_pairs], dtype=int)

            if len(od_ids) <= 1:
                cluster_indexes.append(0.0)
                continue

            sub = S[np.ix_(od_ids, od_ids)]
            sim_intra_C = np.triu(sub, k=1).sum()
            sim_intra_global += sim_intra_C

            outside = np.setdiff1d(all_ids, od_ids, assume_unique=True)
            if outside.size > 0:
                sim_cross_C = S[np.ix_(od_ids, outside)].sum()
            else:
                sim_cross_C = 0.0

            sim_tot_C = sim_intra_C + sim_cross_C
            R_C = 0.0 if sim_tot_C == 0 else float(sim_intra_C / sim_tot_C)

            cluster_indexes.append(R_C)

        self.similarity_index = float(sim_intra_global / sim_total)
        self.cluster_similarity_indexes = cluster_indexes


    def _compute_cluster_congestion_indexes(self):
        n_side = self.grid_side

        congestion_indexes: list[float] = []  # E_c
        congestion_abs: list[float] = []  # E_abs
        congestion_r_max: list[float] = []  # R_max

        for cluster in self.clusters:
            n_agents_cluster = cluster.n_agents

            occ: dict[tuple[int, int], float] = {}

            for od in cluster.od_pairs:
                n_agents_od = len(od.agents)

                paths = od.all_paths
                n_paths = len(paths)

                visit_counts: dict[tuple[int, int], int] = defaultdict(int)
                for path in paths:
                    enc = path.encoded
                    for t, node_id in enumerate(enc):
                        key = (t, int(node_id))
                        visit_counts[key] += 1

                for (t, node_id), count_paths in visit_counts.items():
                    frac = count_paths / n_paths
                    key = (t, node_id)
                    occ[key] = occ.get(key, 0.0) + n_agents_od * frac

            excess_sum = 0.0
            r_max = 0.0

            for (t, node_id), occ_val in occ.items():
                i = node_id // n_side
                j = node_id % n_side
                v = (i, j)

                cap = float(self.G.nodes[v].get("capacity", 0))

                ratio = occ_val / cap

                if ratio > r_max:
                    r_max = ratio

                if occ_val > cap:
                    excess_sum += (occ_val - cap)

            E_abs = excess_sum
            E_c = (E_abs / n_agents_cluster) if n_agents_cluster > 0 else 0.0

            congestion_abs.append(E_abs)
            congestion_indexes.append(E_c)
            congestion_r_max.append(r_max)

        self.cluster_congestion_indexes = congestion_indexes
        self.cluster_congestion_indexes_absolute = congestion_abs
        self.cluster_congestion_ratio_max = congestion_r_max


