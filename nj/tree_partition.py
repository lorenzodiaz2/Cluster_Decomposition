from typing import List

from elements.cluster import Cluster
from nj.nj_torch import NjTree, Node


class TreePartition(NjTree):

    def __init__(self, similarity_matrix, od_pairs, max_cluster_size):
        paths = {od_pair.id: len(od_pair.agents) for od_pair in od_pairs}
        self.od_pairs = od_pairs
        super().__init__(similarity_matrix, paths, max_cluster_size)

    def get_min_cherry(self):
        min_cherry = self.cherries[0]
        for cherry in self.cherries:
            if cherry.n_paths < min_cherry.n_paths:
                min_cherry = cherry
        return min_cherry

    def update_and_prune_leaves(self, cherry):

        leaves = []
        if self.root.id == cherry.id:
            parent = cherry.get_parent()
            self.root = parent

        for link in cherry.links:
            if link.is_leaf:
                self.clusters.remove(link)
                leaves.append(link)

        for leaf in leaves:
            cherry.links.remove(leaf)

        self.clusters.append(cherry)
        self.n_nodes -= 2

    def prune(self, cherry):

        parent = cherry.get_parent()
        links = []
        for link in parent.links:
            if link.id != cherry.id:
                links.append(link)

        # reattach links
        for i in range(3):
            if i < len(links[0].links) and links[0].links[i].id == parent.id:
                links[0].links[i] = links[1]
            if i < len(links[1].links) and links[1].links[i].id == parent.id:
                links[1].links[i] = links[0]

        for link in links:
            if sum(l.is_leaf for l in link.links) == 2:
                if not link.is_cherry:
                    link.is_cherry = True
                    self.cherries.append(link)
        if self.root.id in [cherry.id, parent.id]:
            self.root = links[1] if links[0].is_leaf else links[0]
        self.n_nodes -= 4

    def compute_clusters(self):

        while self.n_nodes >= 5:

            cherry: Node = self.get_min_cherry()
            n_paths = cherry.compute_paths()
            if n_paths < self.max_cluster_size:
                cherry.merge(n_paths)
                self.cherries.remove(cherry)
                self.update_and_prune_leaves(cherry)
                parent = cherry.get_parent()

                if sum(link.is_leaf for link in parent.links) == 2:
                    parent.is_cherry = True
                    self.cherries.append(parent)
            else:
                self.prune(cherry)
                self.cherries.remove(cherry)

            # self.draw()

        self.last_step(self.root)
        self.clusters = sorted(self.clusters, key=lambda x: x.n_paths, reverse=True)

        clusters_list = sorted({
            Cluster(idx, [pair for pair in self.od_pairs if pair.id in cluster_node.cluster])
            for idx, cluster_node in enumerate(self.clusters)
        }, key=lambda c: c.n_agents, reverse=True)

        # self.merge_lightest(clusters_list)
        return clusters_list

    def merge_lightest(self, clusters_list: List[Cluster]):
        last_but_one, last = clusters_list[-2:]
        while last.n_agents + last_but_one.n_agents <= self.max_cluster_size:
            last_but_one.merge(last)
            clusters_list.remove(last)
            last_but_one, last = clusters_list[-2:]

    def last_step(self, root: Node):
        nodes = sorted(root.links, key=lambda node: node.n_paths)
        if len(nodes) == 3:
            n_paths = nodes[0].n_paths + nodes[1].n_paths

            if n_paths < self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(nodes[1])
                root.n_paths = n_paths
                root.cluster += nodes[0].cluster + nodes[1].cluster
                self.clusters.append(root)

                if len(nodes) > 2:
                    n_paths = root.n_paths + nodes[2].n_paths
                    if n_paths < self.max_cluster_size:
                        root.n_paths = n_paths
                        root.cluster += nodes[2].cluster
                        self.clusters.remove(nodes[2])  ############################
        else:
            n_paths = root.n_paths + nodes[0].n_paths
            if n_paths < self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(root)
                root.n_paths = n_paths
                root.cluster += nodes[0].cluster
                self.clusters.append(root)
