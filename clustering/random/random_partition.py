import random


class RandomCluster:
    def __init__(self):
        self.cluster = []
        self.mass = 0

    def add(self, idx, weight):
        self.cluster.append(idx)
        self.mass += weight


class RandomPartition:
    def __init__(self, similarity_matrix, weights, max_cluster_size, seed=None):
        self.n_elements = similarity_matrix.shape[0]
        self.weights = weights
        self.max_cluster_size = max_cluster_size
        self.rng = random.Random(seed)
        self.clusters: list[RandomCluster] = []


    def compute_clusters(self):
        order = list(range(self.n_elements))
        self.rng.shuffle(order)

        current = RandomCluster()
        for idx in order:
            w = self.weights[idx]
            if current.cluster and current.mass + w > self.max_cluster_size:
                self.clusters.append(current)
                current = RandomCluster()
            current.add(idx, w)

        if current.cluster:
            self.clusters.append(current)

        self.clusters = sorted(self.clusters, key=lambda x: x.mass, reverse=True)

