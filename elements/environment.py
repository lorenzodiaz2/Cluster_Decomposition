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
        seed: int | None = 42
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

        self._set_environment()


    def _set_environment(self):
        start = time.time()
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
            initargs=(self.G, self.k),
        ) as pool:
            results = pool.map(compute_paths_for_quadrant, tasks)

        self.od_pairs = [od for group in results for od in group]


        self.agents = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.set_time = time.time() - start


    def compute_clusters(self):
        start = time.time()
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

        self.matrix_time = time.time() - start

        start = time.time()
        tree = TreePartition(self.similarity_matrix, self.od_pairs, self.max_cluster_size)
        self.clusters = tree.compute_clusters()
        self.nj_time = time.time() - start



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
        self._set_quadrants()

        i = 0
        n_tot_agents = 0
        seen = set()
        for quadrant in self.quadrants:
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

        number_of_agents = self.rng.randrange(5, 6)
        agents = [Agent(i, src, dst) for i in range(n_tot_agents, n_tot_agents + number_of_agents)]

        return OD_Pair(id_pair, src, dst, agents)



    def _set_quadrants(self):
        n = self.grid_side
        left_up = (0, 0)
        right_down = (n - 1, n - 1)

        off = self.offset

        if self.n_quadrants == 2:
            self.quadrants = self.divide_by_2(left_up, right_down, off)
        elif self.n_quadrants == 3:
            self.quadrants = self.divide_by_3(left_up, right_down, off)
        elif self.n_quadrants >= 4:
            q = self.n_quadrants // 4
            r = self.n_quadrants - q * 4

            self.quadrants = self.divide_by_4(left_up, right_down, off)

            if q == 1:
                for i in range(r):
                    lu, rd = self.quadrants[i]
                    self.quadrants.extend(self.divide_by_2(lu, rd, off))
                del self.quadrants[:r]

            if q == 2:
                for i in range(4):
                    lu, rd = self.quadrants[i]
                    if i < r:
                        self.quadrants.extend(self.divide_by_3(lu, rd, off))
                    else:
                        self.quadrants.extend(self.divide_by_2(lu, rd, off))
                del self.quadrants[:4]

            if q == 3:
                for i in range(4):
                    lu, rd = self.quadrants[i]
                    if i < r:
                        self.quadrants.extend(self.divide_by_4(lu, rd, off))
                    else:
                        self.quadrants.extend(self.divide_by_3(lu, rd, off))
                del self.quadrants[:4]

            if q == 4 and r == 0:
                for i in range(4):
                    lu, rd = self.quadrants[i]
                    self.quadrants.extend(self.divide_by_4(lu, rd, off))
                del self.quadrants[:4]



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


    def _subgraph_for_quadrant(self, quadrant: Quadrant) -> nx.Graph:
        (top, left), (bottom, right) = quadrant
        nodes = [(i, j) for i in range(top, bottom + 1) for j in range(left, right + 1)]

        return self.G.subgraph(nodes).copy()



    @staticmethod
    def divide_by_2(
        left_up: Coord,
        right_down: Coord,
        offset: int = 0
    ) -> list[Quadrant]:

        top, left = left_up
        bottom, right = right_down

        # split sulle colonne
        left_end_col, right_start_col = Environment._split_interval(left, right, offset)

        left_quadrant: Quadrant = (left_up, (bottom, left_end_col))
        right_quadrant: Quadrant = ((top, right_start_col), right_down)

        return [left_quadrant, right_quadrant]

    @staticmethod
    def divide_by_3(
        left_up: Coord,
        right_down: Coord,
        offset: int = 0
    ) -> list[Quadrant]:
        # prima split orizzontale (colonne) in 2
        left_quadrant, right_quadrant = Environment.divide_by_2(left_up, right_down, offset)
        # poi split verticale (righe) del quadrante di sinistra
        left_up_quadrant, left_down_quadrant = Environment.divide(left_quadrant, offset)

        return [left_up_quadrant, left_down_quadrant, right_quadrant]


    @staticmethod
    def divide_by_4(
        left_up: Coord,
        right_down: Coord,
        offset: int = 0
    ) -> list[Quadrant]:
        # prima divido in 3 (left_up, left_down, right intero)
        left_up_quadrant, left_down_quadrant, right_quadrant = Environment.divide_by_3(
            left_up, right_down, offset
        )
        # poi split verticale del quadrante di destra
        right_up_quadrant, right_down_quadrant = Environment.divide(right_quadrant, offset)

        return [left_up_quadrant, left_down_quadrant, right_up_quadrant, right_down_quadrant]


    @staticmethod
    def divide(
        quadrant: Quadrant,
        offset: int = 0
    ) -> tuple[Quadrant, Quadrant]:
        (top, left), (bottom, right) = quadrant

        # split sulle righe
        up_bottom_row, down_top_row = Environment._split_interval(top, bottom, offset)

        up_quadrant: Quadrant = ((top, left), (up_bottom_row, right))
        down_quadrant: Quadrant = ((down_top_row, left), (bottom, right))

        return up_quadrant, down_quadrant


    @staticmethod
    def _split_interval(start: int, end: int, offset: int) -> tuple[int, int]:
        """
        Divide [start, end] in due intervalli:
            sinistra:  [start, left_end]
            destra:    [right_start, end]
        con comportamento:

        - offset =  0 -> 1 cella in comune (mid)
        - offset >  0 -> entrambi si espandono verso il centro (più overlap)
        - offset <  0 -> entrambi si restringono verso l'esterno (buco in mezzo)

        Esempio: start=0, end=11 (12 celle), mid=5
            offset=0: left=[0..5], right=[5..11]
            offset=1: left=[0..6], right=[4..11]
            offset=-1: left=[0..4], right=[6..11]
        """
        mid = (start + end) // 2

        # espansione/restringimento simmetrico attorno a mid
        left_end = mid + offset
        right_start = mid - offset

        # clamp per rimanere dentro [start, end]
        left_end = max(start, min(left_end, end))
        right_start = max(start, min(right_start, end))

        return left_end, right_start

