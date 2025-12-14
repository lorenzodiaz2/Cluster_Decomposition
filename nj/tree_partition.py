from typing import List

from elements.cluster import Cluster
from nj.nj_torch import NjTree, Node


class TreePartition(NjTree):

    def __init__(self, similarity_matrix, od_pairs, max_cluster_size):
        n_agents = {i: len(od_pairs[i].agents) for i in range(len(od_pairs))}
        self.od_pairs = od_pairs
        super().__init__(similarity_matrix, n_agents, max_cluster_size)

    # def get_min_cherry(self):
    #     min_cherry = self.cherries[0]
    #     for cherry in self.cherries:
    #         if cherry.n_agents < min_cherry.n_agents:
    #             min_cherry = cherry
    #     return min_cherry

    def get_min_cherry(self):
        return min(self.cherries, key=lambda ch: ch.compute_n_agents())

    def update_and_prune_leaves(self, cherry):

        leaves = []
        if self.root.id == cherry.id:
            parent = cherry.get_parent()
            self.root = parent

        for link in cherry.adj_nodes:
            if link.is_leaf:
                self.clusters.remove(link)
                leaves.append(link)

        for leaf in leaves:
            cherry.adj_nodes.remove(leaf)

        self.clusters.append(cherry)
        self.n_nodes -= 2

    def prune(self, cherry):

        parent = cherry.get_parent()
        links = []
        for link in parent.adj_nodes:
            if link.id != cherry.id:
                links.append(link)

        # reattach links
        for i in range(3):
            if i < len(links[0].adj_nodes) and links[0].adj_nodes[i].id == parent.id:
                links[0].adj_nodes[i] = links[1]
            if i < len(links[1].adj_nodes) and links[1].adj_nodes[i].id == parent.id:
                links[1].adj_nodes[i] = links[0]

        for link in links:
            if sum(l.is_leaf for l in link.adj_nodes) == 2:
                if not link.is_cherry:
                    link.is_cherry = True
                    self.cherries.append(link)
        if self.root.id in [cherry.id, parent.id]:
            self.root = links[1] if links[0].is_leaf else links[0]
        self.n_nodes -= 4

    def compute_clusters(self):

        while self.n_nodes >= 5:

            cherry: Node = self.get_min_cherry()
            n_agents = cherry.compute_n_agents()
            if n_agents <= self.max_cluster_size:
                cherry.merge(n_agents)
                self.cherries.remove(cherry)
                self.update_and_prune_leaves(cherry)
                parent = cherry.get_parent()

                if sum(link.is_leaf for link in parent.adj_nodes) == 2:
                    parent.is_cherry = True
                    self.cherries.append(parent)
            else:
                self.prune(cherry)
                self.cherries.remove(cherry)

            # self.draw()

        self.last_step(self.root)
        self.clusters = sorted(self.clusters, key=lambda x: x.n_agents, reverse=True)

        # clusters_list = [Cluster(idx, [pair for pair in self.od_pairs if pair.id in cluster_node.cluster]) for idx, cluster_node in enumerate(self.clusters)]
        clusters_list = [
            Cluster(idx, [self.od_pairs[i] for i in cluster_node.cluster])
            for idx, cluster_node in enumerate(self.clusters)
        ]
        self.merge_lightest(clusters_list)
        return clusters_list

    def merge_lightest(self, clusters_list: List[Cluster]):
        last_but_one, last = clusters_list[-2:]
        while last.n_agents + last_but_one.n_agents <= self.max_cluster_size:
            last_but_one.merge(last)
            clusters_list.remove(last)
            last_but_one, last = clusters_list[-2:]

    def last_step(self, root: Node):
        nodes = sorted(root.adj_nodes, key=lambda node: node.n_agents)
        if len(nodes) == 3:
            n_agents = nodes[0].n_agents + nodes[1].n_agents

            if n_agents < self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(nodes[1])
                root.n_agents = n_agents
                root.cluster += nodes[0].cluster + nodes[1].cluster
                self.clusters.append(root)

                if len(nodes) > 2:
                    n_agents = root.n_agents + nodes[2].n_agents
                    if n_agents < self.max_cluster_size:
                        root.n_agents = n_agents
                        root.cluster += nodes[2].cluster
                        self.clusters.remove(nodes[2])  ############################
        else:
            n_agents = root.n_agents + nodes[0].n_agents
            if n_agents < self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(root)
                root.n_agents = n_agents
                root.cluster += nodes[0].cluster
                self.clusters.append(root)
