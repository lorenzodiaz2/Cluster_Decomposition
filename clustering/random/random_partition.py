import math
import random


class RandomCluster:
    def __init__(self):
        self.cluster = []
        self.n_agents = 0

    def add(self, idx, n_agents):
        self.cluster.append(idx)
        self.n_agents += n_agents


class RandomPartition:
    def __init__(self, similarity_matrix, n_agents, max_cluster_size, seed=None):
        self.n_taxa = similarity_matrix.shape[0]
        self.n_agents = n_agents
        self.max_cluster_size = max_cluster_size
        self.rng = random.Random(seed)
        self.clusters: list[RandomCluster] = []


    def compute_clusters(self):
        order = list(range(self.n_taxa))
        self.rng.shuffle(order)

        current = RandomCluster()
        for idx in order:
            w = self.n_agents[idx]
            if current.cluster and current.n_agents + w > self.max_cluster_size:
                self.clusters.append(current)
                current = RandomCluster()
            current.add(idx, w)

        if current.cluster:
            self.clusters.append(current)

        self.clusters = sorted(self.clusters, key=lambda x: x.n_agents, reverse=True)

