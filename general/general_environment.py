import random
import time
from typing import Callable, Hashable, Optional, TypeVar, Any
from abc import ABC, abstractmethod

import multiprocessing as mp
import networkx as nx
import numpy as np

from general.general_cluster import General_Cluster
from general.nj.tree_partition import TreePartition
from utils.grid_utils import set_replicated_quadrants
from utils.parallel import init_similarity, compute_similarity_row

Coord = tuple[int, int]
Quadrant = tuple[Coord, Coord]
T = TypeVar("T")


class General_Environment(ABC):
    def __init__(
        self,
        grid_side: int,
        max_cluster_size: int,
        n_quadrants: int,
        n_elements_per_quadrant: int,
        offset: int,
        k: int,
        reproducibility_flag: bool,
        remove_rnd_nodes_flag: bool,
        remove_percentage: float,
        seed: int,
        instance_file: str
    ):
        self.grid_side = grid_side
        self.max_cluster_size = max_cluster_size
        self.n_quadrants = n_quadrants
        self.n_elements_per_quadrant = n_elements_per_quadrant
        self.offset = offset
        self.k = k
        self.G = None
        self.elements = []
        self.clusters: list[General_Cluster] = []
        self.similarity_matrix = None
        self.set_time = None
        self.matrix_time = None
        self.nj_time = None
        self.rng = random.Random(seed if reproducibility_flag else None)
        self.remove_rnd_nodes_flag = remove_rnd_nodes_flag
        self.remove_percentage = remove_percentage
        self.quadrants: list[Quadrant] = []
        self.quadrant_by_element_id: dict[int, Quadrant] = {}
        self.clusters_solvers = []
        self.instance_file = instance_file

        self.similarity_index = None                     # frazione della similarità totale che resta dentro i cluster
        self.cluster_similarity_indexes = None           # per ogni cluster C, quota della similarità “che coinvolge C” che è interna a C

        self.cluster_congestion_indexes = None           # quanto il cluster è “stretto” rispetto alle capacità
        self.cluster_congestion_indexes_absolute = None  # quanta capacità manca in totale nel cluster
        self.cluster_congestion_ratio_max = None         # il worst bottleneck del cluster

        self.global_congestion_index_absolute = None     # totale overload sulle risorse quando considero tutti gli element insieme
        self.global_congestion_ratio_max = None          # il peggior collo di bottiglia dell’intera istanza

        self.cross_congestion_index_absolute = None      # quanta congestione nasce solo perché cluster diversi condividono le stesse risorse
        self.cross_congestion_rate = None                # quanta congestione inter-cluster c’è per agente (o altra cosa se per altri problemi)
        self.cross_congestion_share = None               # di tutta la congestione che c’è nel completo, quanta è colpa dell’interazione tra cluster

        self._cluster_elements_index = None

        # mi servono come cache per il calcolo degli indici di congestione
        self._cluster_occ_cache = None
        self._global_occ_cache = None
        self._cluster_mass_cache = None
        self._global_mass_cache = None


    def _set_environment(self):
        start = time.perf_counter()
        if self.instance_file:
            self._retrieve_instance()
        else:
            self._create_grid_graph()
            self._compute_quadrants()
            self._choose_elements()
        self.set_time = time.perf_counter() - start



    def _compute_quadrants(self):
        if self.n_quadrants == 1:
            self.quadrants = [((0, 0), (self.grid_side - 1, self.grid_side - 1))]
        else:
            self.quadrants = set_replicated_quadrants(self.grid_side, self.n_quadrants, self.offset)



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
            element_ids = np.array([self._cluster_elements_index[element.id] for element in cluster.elements], dtype=int)

            if len(element_ids) <= 1:
                cluster_indexes.append(0.0)
                continue

            sub = S[np.ix_(element_ids, element_ids)]
            sim_intra_C = np.triu(sub, k=1).sum()
            sim_intra_global += sim_intra_C

            outside = np.setdiff1d(all_ids, element_ids, assume_unique=True)
            if outside.size > 0:
                sim_cross_C = S[np.ix_(element_ids, outside)].sum()
            else:
                sim_cross_C = 0.0

            sim_tot_C = sim_intra_C + sim_cross_C
            R_C = 0.0 if sim_tot_C == 0 else float(sim_intra_C / sim_tot_C)

            cluster_indexes.append(R_C)

        self.similarity_index = float(sim_intra_global / sim_total)
        self.cluster_similarity_indexes = cluster_indexes


    def compute_clusters(
        self,
        sim_fn: Callable
    ):
        start = time.perf_counter()
        n = len(self.elements)
        self.similarity_matrix = np.zeros((n, n), dtype=int)
        self._cluster_elements_index = {el.id: i for i, el in enumerate(self.elements)}

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_similarity,
            initargs=(self.elements, sim_fn),
        ) as pool:
            results = pool.map(compute_similarity_row, range(n - 1))

        for i, row in results:
            self.similarity_matrix[i, i + 1:] = row[i + 1:]
            self.similarity_matrix[i + 1:, i] = row[i + 1:]

        self.matrix_time = time.perf_counter() - start

        start = time.perf_counter()
        weight_for_element = self._build_weight()
        tree = TreePartition(self.similarity_matrix, weight_for_element, self.max_cluster_size)
        tree.compute_clusters()

        self._build_clusters(tree)

        self.nj_time = time.perf_counter() - start
        self._compute_congestion_indexes()
        self._compute_similarity_index()


    def _merge(self, clusters, condition_fn: Callable):
        while len(clusters) >= 2:
            last_but_one = clusters[-2]
            last = clusters[-1]
            if condition_fn(last_but_one, last) > self.max_cluster_size:
                break
            last_but_one.merge(last)
            clusters.pop()


    def _spawn_in_quadrants(
        self,
        n_per_quadrant: int,
        maker: Callable[[Quadrant, int], T],
        seen_key: Callable[[T], Hashable],
        target_list: list[T],
        quadrant_by_id: dict[int, Quadrant],
        start_id: int = 0,
        seen: Optional[set[Hashable]] = None,
        id_getter: Callable[[T], int] = lambda x: x.id,
    ) -> int:

        if seen is None:
            seen = set()

        cur_id = start_id

        for j, quadrant in enumerate(self.quadrants):
            if j >= self.n_quadrants:
                break

            for _ in range(n_per_quadrant):
                while True:
                    obj = maker(quadrant, cur_id)
                    key = seen_key(obj)
                    if key in seen:
                        continue
                    seen.add(key)
                    break

                target_list.append(obj)
                quadrant_by_id[id_getter(obj)] = quadrant
                cur_id += 1

        return cur_id


    def _solve_clusters(
        self,
        sim_fn: Callable,
        solver_factory: Callable,
        post_fn: Callable | None = None
    ):
        self.compute_clusters(sim_fn)
        self.clusters_solvers = []

        for cluster in self.clusters:
            hs = solver_factory(cluster)
            hs.solve()
            self.clusters_solvers.append(hs)

        if post_fn is not None:
            post_fn()


    def _invalidate_expected_occupancy_cache(self) -> None:
        self._cluster_occ_cache = None
        self._global_occ_cache = None
        self._cluster_mass_cache = None
        self._global_mass_cache = None


    def _compute_congestion_indexes(self) -> None:
        self._invalidate_expected_occupancy_cache()
        self._ensure_expected_occupancy_cache()

        cluster_occ = self._cluster_occ_cache or []
        cluster_mass = self._cluster_mass_cache or []

        cong, cong_abs, cong_rmax = [], [], []
        for occ_c, mass_c in zip(cluster_occ, cluster_mass):
            E_abs, R_max = self._congestion_from_occ(occ_c)
            E = (E_abs / mass_c) if mass_c > 0 else 0.0
            cong.append(E); cong_abs.append(E_abs); cong_rmax.append(R_max)

        self.cluster_congestion_indexes = cong
        self.cluster_congestion_indexes_absolute = cong_abs
        self.cluster_congestion_ratio_max = cong_rmax

        E_abs_g, R_max_g = self._congestion_from_occ(self._global_occ_cache or {})
        self.global_congestion_index_absolute = E_abs_g
        self.global_congestion_ratio_max = R_max_g

        E_cross = max(0.0, E_abs_g - float(sum(cong_abs)))
        denominator = float(self._global_mass_cache or 0.0)
        self.cross_congestion_index_absolute = E_cross
        self.cross_congestion_rate = (E_cross / denominator) if denominator > 0 else 0.0
        self.cross_congestion_share = E_cross / (E_abs_g + 1e-9)

    @abstractmethod
    def _ensure_expected_occupancy_cache(self) -> None:
        """calcola le quattro cache"""
        raise NotImplementedError

    @abstractmethod
    def _congestion_from_occ(self, occ: dict[Any, float]) -> tuple[float, float]:
        raise NotImplementedError


    @abstractmethod
    def _build_weight(self):
        raise NotImplementedError

    @abstractmethod
    def _build_clusters(self, tree: TreePartition):
        raise NotImplementedError

    @abstractmethod
    def _choose_elements(self):
        raise NotImplementedError

    @abstractmethod
    def _retrieve_instance(self):
        raise NotImplementedError

