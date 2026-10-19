import random
import time
from typing import Callable, Hashable, Optional, TypeVar, Any
from abc import ABC, abstractmethod

import multiprocessing as mp
import networkx as nx
import numpy as np

from caps.general.general_cluster import General_Cluster
from clustering.nj.tree_partition import TreePartition
from clustering.random.random_partition import RandomPartition
from utils.grid_utils import set_replicated_quadrants, walls_with_gates, city_blocks
from utils.parallel import init_similarity, compute_similarity_row

Coord = tuple[int, int]
Quadrant = tuple[Coord, Coord]
T = TypeVar("T")


class General_Environment(ABC):
    def __init__(
        self,
        grid_side: int | None = None,
        max_cluster_size: int | None = None,
        n_quadrants: int | None = None,
        n_elements_per_quadrant: int | None = None,
        resources_capacity: int | None = None,
        offset: int | None = None,
        k: int | None = None,
        reproducibility_flag: bool | None = None,
        remove_rnd_nodes_flag: bool | None = None,
        remove_percentage: float | None = None,
        seed: int | None = None,
        sim_method: tuple[str, float | None] | None = ("uniform", None),
        resource_weights: str = "linear",
        instance_file: str | None = None,
        partitioner: str = "nj",
        family: str = "quadrants",
        family_params: dict | None = None
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
        self.partition_time = None
        self.seed = seed if reproducibility_flag else None
        self.rng = random.Random(self.seed)
        self.partitioner = partitioner                   # "nj" oppure "random"
        self.family = family                             # "quadrants", "uniform", "hotspots"
        self.family_params = family_params or {}         # parametri della famiglia (L_min, L_max, n_hotspots, sigma, noise, obstacles, ...)
        self.remove_rnd_nodes_flag = remove_rnd_nodes_flag
        self.remove_percentage = remove_percentage
        self.quadrants: list[Quadrant] = []
        self.quadrant_by_element_id: dict[int, Quadrant] = {}
        self.clusters_solvers = []
        self.instance_file = instance_file
        self.resources_capacity = resources_capacity
        self.sim_method = sim_method
        self.resource_weights = resource_weights

        self.similarity_index = None                     # frazione della similarità totale che resta dentro i cluster
        self.cluster_similarity_indexes = None           # per ogni cluster C, quota della similarità “che coinvolge C” che è interna a C

        self.cluster_congestion_indexes = None           # quanto il cluster è “stretto” rispetto alle capacità
        self.cluster_congestion_indexes_absolute = None  # quanta capacità manca in totale nel cluster
        self.cluster_congestion_ratio_max = None         # il worst bottleneck del cluster

        self.global_congestion_index_absolute = None     # totale overload sulle risorse quando considero tutti gli element insieme
        self.global_congestion_ratio_max = None          # il peggior collo di bottiglia dell’intera istanza

        self.cross_congestion_index_absolute = None      # quanta congestione nasce solo perché cluster diversi condividono le stesse risorse
        self.cross_congestion_rate = None                # quanta congestione inter-cluster c’è per unità di massa (somma dei pesi delle entità)
        self.cross_congestion_share = None               # di tutta la congestione che c’è nel completo, quanta è colpa dell’interazione tra cluster

        self._cluster_elements_index = None

        self.max_mean_intercluster_similarity = None
        self.silhouette_score = None
        self.element_silhouette_scores = None

        self.normalized_macro_components = None          # Valore Z/N
        self.max_eigengap = None                         # Salto massimo nello spettro

        self.clustering_distortion_ratio = None          # K_expected / K_actual

        # --- Metriche Topologiche ---
        self.interaction_density = None  # D_sim
        self.similarity_gini = None  # G_sim
        self.similarity_skewness = None  # gamma_sim

        # --- Metriche di Partizione ---
        self.mass_cv = None  # CV_mass
        self.boundary_entities_ratio = None  # R_boundary
        self.partition_entropy = None  # H_P

        # --- Metriche focalizzate sulle Risorse ---
        self.critical_resources_ratio = None  # R_crit
        self.congestion_gini = None  # G_occ
        self.cross_cluster_footprint_density = None  # D_cross_res

        # --- Metriche di Flessibilità ---
        self.average_footprint_breadth = None  # B_opt
        self.average_guaranteed_slack = None  # F_slack
        self.space_to_mass_ratio = None  # R_SM

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

        n = self.grid_side if self.grid_side else 0
        self.G.add_nodes_from((i, j) for i in range(n) for j in range(n))
        self.G.add_edges_from([((i, j), (i + 1, j)) for i in range(n - 1) for j in range(n)] +
                         [((i, j), (i, j + 1)) for i in range(n) for j in range(n - 1)])


        removed = set()
        if self.remove_rnd_nodes_flag:
            removed |= {v for v in self.G.nodes() if self.rng.random() < self.remove_percentage}

        obstacles = self.family_params.get("obstacles")
        if obstacles == "walls":
            removed |= walls_with_gates(n, self.family_params.get("n_walls", 2), self.family_params.get("n_gates", 1), self.rng)
        elif obstacles == "blocks":
            removed |= city_blocks(n, self.family_params.get("block_size", 3), self.family_params.get("street_width", 1))
        elif obstacles is not None:
            raise ValueError(f"obstacles '{obstacles}' non supportato")

        if removed:
            self.G.remove_nodes_from(removed)
            # tengo solo la componente connessa più grande, così ogni coppia di nodi rimasti è collegata
            if self.G.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(self.G), key=len)
                self.G.remove_nodes_from([v for v in list(self.G.nodes) if v not in largest_cc])


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
        sim_fn: Callable,
        resource_weights: dict | None = None
    ):
        start = time.perf_counter()
        n = len(self.elements)
        self.similarity_matrix = np.zeros((n, n), dtype=float)
        self._cluster_elements_index = {el.id: i for i, el in enumerate(self.elements)}

        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_similarity,
            initargs=(self.elements, sim_fn, resource_weights),
        ) as pool:
            results = pool.map(compute_similarity_row, range(n - 1))

        for i, row in results:
            self.similarity_matrix[i, i + 1:] = row[i + 1:]
            self.similarity_matrix[i + 1:, i] = row[i + 1:]

        self.matrix_time = time.perf_counter() - start

        start = time.perf_counter()
        weight_for_element = self._build_weight()
        if self.partitioner == "nj":
            tree = TreePartition(self.similarity_matrix, weight_for_element, self.max_cluster_size)
        elif self.partitioner == "random":
            tree = RandomPartition(self.similarity_matrix, weight_for_element, self.max_cluster_size, seed=self.seed)
        else:
            raise ValueError(f"partitioner '{self.partitioner}' non supportato")
        tree.compute_clusters()

        self._build_clusters(tree)

        self.partition_time = time.perf_counter() - start
        self._compute_congestion_indexes()
        self._compute_similarity_index()
        self._compute_max_mean_intercluster_similarity()
        self._compute_silhouette_score()
        self._compute_spectral_metrics()
        self._compute_expected_clusters()
        self.compute_metrics()


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
        solver_maker: Callable,
        post_fn: Callable | None = None,
        resource_weights: dict | None = None
    ):
        self.compute_clusters(sim_fn, resource_weights)
        self.clusters_solvers = []

        for cluster in self.clusters:
            hs = solver_maker(self, cluster)
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
            cong.append(E)
            cong_abs.append(E_abs)
            cong_rmax.append(R_max)

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


    def _cluster_element_indices(self, cluster) -> np.ndarray:
        # mappa element.id -> indice di riga/colonna in S
        return np.array(
            [self._cluster_elements_index[e.id] for e in cluster.elements],
            dtype=int
        )

    def _compute_max_mean_intercluster_similarity(self) -> None:
        """
        Ritorna:
          - max_mean_sim: massimo della similarità media tra coppie di cluster (A != B)
          - argmax_pair: (idx_cluster_A, idx_cluster_B) che realizza il massimo, oppure None se non definito
        """
        S = self.similarity_matrix
        if S is None or self.clusters is None or len(self.clusters) < 2:
            return

        # Pre calcolo indici elementi per cluster
        cluster_ids = [self._cluster_element_indices(c) for c in self.clusters]
        sizes = [len(ids) for ids in cluster_ids]

        max_mean = 0.0

        for a in range(len(cluster_ids) - 1):
            Ia = cluster_ids[a]
            na = sizes[a]
            if na == 0:
                continue

            for b in range(a + 1, len(cluster_ids)):
                Ib = cluster_ids[b]
                nb = sizes[b]
                if nb == 0:
                    continue

                cross_sum = float(S[np.ix_(Ia, Ib)].sum())
                mean_sim = cross_sum / (na * nb)

                if mean_sim > max_mean:
                    max_mean = mean_sim

        self.max_mean_intercluster_similarity = max_mean



    def _build_cluster_labels(self) -> np.ndarray:
        """
        Ritorna un array labels di lunghezza n (n = #elements),
        dove labels[i] = indice del cluster a cui appartiene l'elemento con indice i nella similarity_matrix.
        """
        n = len(self.elements)
        labels = np.full(n, -1, dtype=int)

        for c_idx, cluster in enumerate(self.clusters):
            for e in cluster.elements:
                labels[self._cluster_elements_index[e.id]] = c_idx

        return labels


    def _compute_silhouette_score(
        self,
        distance_eps: float = 1e-12,
    ) -> None:
        S = self.similarity_matrix
        if S is None or self.clusters is None or len(self.clusters) < 2:
            self.silhouette_score = 0.0
            self.element_silhouette_scores = None
            return

        n = S.shape[0]
        labels = self._build_cluster_labels()

        cluster_ids = [self._cluster_element_indices(c) for c in self.clusters]
        cluster_sizes = np.array([len(ids) for ids in cluster_ids], dtype=int)

        if np.sum(cluster_sizes > 0) < 2:
            self.silhouette_score = 0.0
            self.element_silhouette_scores = None
            return

        Smax = float(S.max()) if n > 0 else 0.0
        if Smax <= 0.0:
            self.silhouette_score = 0.0
            self.element_silhouette_scores = np.zeros(n, dtype=float)
            return


        D = 1.0 - S / Smax  # distanza in [0,1], adimensionale
        np.fill_diagonal(D, 0.0)

        s = np.zeros(n, dtype=float)  # singoletti -> s = 0 (definizione standard)

        for i in range(n):
            ci = labels[i]
            ids_ci = cluster_ids[ci]
            if len(ids_ci) <= 1:
                continue

            a_i = D[i, ids_ci].sum() / (len(ids_ci) - 1)  # D[i,i] = 0, quindi è già escluso

            others = [D[i, ids].mean() for c, ids in enumerate(cluster_ids)
                      if c != ci and len(ids) > 0]
            if not others:
                continue
            b_i = min(others)

            denom = max(a_i, b_i)
            s[i] = 0.0 if denom <= distance_eps else (b_i - a_i) / denom

        self.element_silhouette_scores = s
        self.silhouette_score = float(s.mean())


    def _compute_spectral_metrics(self, zero_tol: float = 1e-9) -> None:
        S = self.similarity_matrix
        if S is None or not self.clusters:
            return

        N = len(self.clusters)

        if N < 2:
            self.normalized_macro_components = 1.0
            self.max_eigengap = 0.0
            return

        W_cond = np.zeros((N, N), dtype=float)
        cluster_ids = [self._cluster_element_indices(c) for c in self.clusters]

        for i in range(N):
            idx_i = cluster_ids[i]
            if len(idx_i) == 0:
                continue
            for j in range(i + 1, N):
                idx_j = cluster_ids[j]
                if len(idx_j) == 0:
                    continue

                cross_sim = float(S[np.ix_(idx_i, idx_j)].sum())
                W_cond[i, j] = cross_sim
                W_cond[j, i] = cross_sim

        L = self._normalized_laplacian(W_cond)
        eigenvalues = np.linalg.eigvalsh(L)

        Z = np.sum(eigenvalues <= zero_tol)
        self.normalized_macro_components = float(Z) / N

        m = min(N, 30)  # solo la parte bassa dello spettro
        eigengaps = np.diff(eigenvalues[:m])
        self.max_eigengap = float(np.max(eigengaps)) if len(eigengaps) > 0 else 0.0


    def _compute_expected_clusters(self, k_max: int = 30) -> None:
        S = self.similarity_matrix
        N = S.shape[0]
        K_actual = len(self.clusters) if self.clusters else 1

        k_max = min(N - 1, max(k_max, 2 * K_actual))
        L = self._normalized_laplacian(S)
        eigenvalues = np.linalg.eigvalsh(L)[:k_max + 1]

        if len(eigenvalues) > 1:
            k_expected = int(np.argmax(np.diff(eigenvalues))) + 1
        else:
            k_expected = 1

        self.clustering_distortion_ratio = float(k_expected) / K_actual

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
    def _build_clusters(self, tree: TreePartition | RandomPartition):
        raise NotImplementedError

    @abstractmethod
    def _choose_elements(self):
        raise NotImplementedError

    @abstractmethod
    def _retrieve_instance(self):
        raise NotImplementedError

    @abstractmethod
    def _get_active_resource_capacities(self) -> dict[Any, float]:
        """Restituisce {r_id: capacità_nominale} per tutte le risorse attive."""
        raise NotImplementedError


    @abstractmethod
    def _compute_flexibility_metrics(self, occ_global: dict[Any, float], cap_dict: dict[Any, float]) -> tuple[
        float, float, float]:
        """Restituisce (B_opt, F_slack, R_SM) analizzando le singole opzioni delle entità."""
        raise NotImplementedError


    def compute_metrics(self) -> None:
        S = self.similarity_matrix
        if S is None or self.clusters is None or not self.clusters:
            return

        n = S.shape[0]
        m = len(self.clusters)

        # 1. Metriche Topologiche
        non_zero_S = S[S > 0]
        if len(non_zero_S) > 0:
            # Interaction Density
            self.interaction_density = len(non_zero_S) / (n * (n - 1)) if n > 1 else 0.0

            # Similarity Skewness
            mean_S = np.mean(non_zero_S)
            std_S = np.std(non_zero_S)
            self.similarity_skewness = float(
                np.mean(((non_zero_S - mean_S) ** 3)) / (std_S ** 3)) if std_S > 1e-9 else 0.0
        else:
            self.interaction_density = 0.0
            self.similarity_skewness = 0.0

        # Similarity Gini Index
        s_degrees = np.sum(S, axis=1)
        s_deg_sorted = np.sort(s_degrees)
        s_deg_sum = np.sum(s_deg_sorted)
        if s_deg_sum > 0:
            index_arr = np.arange(1, n + 1)
            self.similarity_gini = float((2.0 * np.sum(index_arr * s_deg_sorted)) / (n * s_deg_sum) - (n + 1.0) / n)
        else:
            self.similarity_gini = 0.0

        # 2. Metriche di Qualità della Partizione
        masses = np.array(self._cluster_mass_cache, dtype=float) if self._cluster_mass_cache else np.zeros(m)
        mass_mean = np.mean(masses)
        self.mass_cv = float(np.std(masses) / mass_mean) if mass_mean > 0 else 0.0

        # Boundary Entities Ratio
        labels = self._build_cluster_labels()
        S_cluster = np.zeros((n, m))
        for k in range(m):
            indices_k = np.where(labels == k)[0]
            if len(indices_k) > 0:
                S_cluster[:, k] = np.sum(S[:, indices_k], axis=1)

        if n > 0:
            own = S_cluster[np.arange(n), labels]
            best = S_cluster.max(axis=1)
            self.boundary_entities_ratio = float(np.sum(best > own + 1e-12) / n)
        else:
            self.boundary_entities_ratio = 0.0

        # Partition Entropy
        mass_tot = np.sum(masses)
        if mass_tot > 0 and m > 1:
            p = masses[masses > 0] / mass_tot
            self.partition_entropy = float(-np.sum(p * np.log2(p)) / np.log2(m))
        else:
            self.partition_entropy = 0.0

        # 3. Metriche focalizzate sulle Risorse
        occ_global = self._global_occ_cache or {}
        cap_dict = self._get_active_resource_capacities()
        active_res = list(occ_global.keys())
        n_active_res = len(active_res)

        if n_active_res > 0:
            n_crit = sum(1 for r in active_res if occ_global[r] > cap_dict.get(r, float('inf')))
            self.critical_resources_ratio = n_crit / n_active_res

            # Congestion Gini Index
            occ_ratios = [occ_global[r] / cap_dict[r] for r in active_res if cap_dict.get(r, 0) > 0]
            if occ_ratios:
                occ_r_sorted = np.sort(occ_ratios)
                n_r = len(occ_r_sorted)
                occ_sum = np.sum(occ_r_sorted)
                if occ_sum > 0:
                    idx_arr = np.arange(1, n_r + 1)
                    self.congestion_gini = float(
                        (2.0 * np.sum(idx_arr * occ_r_sorted)) / (n_r * occ_sum) - (n_r + 1.0) / n_r)
                else:
                    self.congestion_gini = 0.0
            else:
                self.congestion_gini = 0.0

            # Cross-Cluster Footprint Density
            n_cross = sum(
                1 for r in active_res if sum(1 for c_occ in self._cluster_occ_cache if c_occ.get(r, 0) > 0) > 1)
            self.cross_cluster_footprint_density = n_cross / n_active_res
        else:
            self.critical_resources_ratio = 0.0
            self.congestion_gini = 0.0
            self.cross_cluster_footprint_density = 0.0

        # 4. Metriche di Flessibilità
        b_opt, f_slack, r_sm = self._compute_flexibility_metrics(occ_global, cap_dict)
        self.average_footprint_breadth = b_opt
        self.average_guaranteed_slack = f_slack
        self.space_to_mass_ratio = r_sm


    @staticmethod
    def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
        deg = W.sum(axis=1)
        inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        L = -(inv_sqrt[:, None] * W * inv_sqrt[None, :])
        L[np.diag_indices_from(L)] += nz.astype(float)  # nodi isolati: riga a 0 (componente a sé)
        return L