import time
from typing import List
import numpy as np

from elements.agent import Agent
from elements.cluster import Cluster
from elements.pair import OD_Pair
from nj.tree_partition import TreePartition
from utils.graph_functions import create_grid_graph, choose_pairs


class Environment:
    def __init__(
        self,
        grid_size: int,
        max_cluster_size: int,
        num_pairs_per_quadrant: int,
        offset: int,
        k: int
    ):
        self.grid_size = grid_size
        self.max_cluster_size = max_cluster_size
        self.num_pairs_per_quadrant = num_pairs_per_quadrant
        self.offset = offset
        self.k = k
        self.G = None
        self.od_pairs: List[OD_Pair] = []
        self.agents: List[Agent] = []
        self.clusters: List[Cluster] = []
        self.T = 0
        self.set_time = None
        self.cluster_time = None

        self.set_environment()


    def set_environment(self):
        start = time.time()
        self.G = create_grid_graph(self.grid_size)
        self.od_pairs = choose_pairs(self.G, self.num_pairs_per_quadrant, self.offset)
        for od_pair in self.od_pairs:
            od_pair.compute_k_shortest_paths(self.G, self.k)
        self.T = max(len(od_pair.k_shortest_paths[self.k - 1].visits) - 1 for od_pair in self.od_pairs)
        for od_pair in self.od_pairs:
            od_pair.delay_shortest_paths(self.T)
        self.agents = [a for od_pair in self.od_pairs for a in od_pair.agents]
        self.set_time = time.time() - start

        print(f"\nGRID SIZE = {self.grid_size}     PAIRS PER QUADRANT = {self.num_pairs_per_quadrant}     MAX CLUSTER SIZE = {self.max_cluster_size}     k = {self.k}     NUMBER OF AGENTS = {len(self.agents)}")
        print(f"\nEnvironment Created     Time: {self.set_time}\n")



    def compute_clusters(self):
        start = time.time()
        similarity_matrix = np.array([[0 if od_pair1.id == od_pair2.id else od_pair1.compute_similarity(od_pair2) for od_pair2 in self.od_pairs] for od_pair1 in self.od_pairs])
        tree = TreePartition(similarity_matrix, self.od_pairs, self.max_cluster_size)
        self.clusters = tree.compute_clusters()
        self.cluster_time = time.time() - start
        print(f"\nCreated {len(self.clusters)} clusters.  Time = {self.cluster_time}")
        for cluster in self.clusters:
            print("    ", cluster)
            # plot_paths(self.G, cluster.od_pairs)
        print()
