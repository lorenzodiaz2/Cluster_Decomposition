import networkx as nx
import torch
from matplotlib import pyplot as plt


class Node:
    def __init__(self, idx, is_leaf=False, is_cherry=False, n_paths=0):
        self.id = idx
        self.is_leaf = is_leaf
        self.is_cherry = is_cherry
        self.links = []
        self.n_paths = n_paths
        self.cluster = [self.id] if is_leaf else []

    def compute_paths(self):
        n_paths = 0
        for link in self.links:
            n_paths += link.n_paths
        return n_paths

    def merge(self, n_paths):
        self.n_paths += n_paths
        for link in self.links:
            if link.is_leaf:
                self.cluster += link.cluster

        self.is_leaf = True
        self.is_cherry = False

    def get_parent(self):
        parent = None
        for link in self.links:
            if not link.is_leaf:
                parent = link
        return parent



class NjTree:
    def __init__(self, similarity_matrix, paths, max_cluster_size):
        self.d = similarity_matrix
        self.n_taxa = similarity_matrix.shape[0]
        self.max_cluster_size = max_cluster_size
        self.cherries = []
        self.n_nodes = 2 * self.n_taxa - 2
        self.paths = paths

        self.nodes = ([Node(i, is_leaf=True, n_paths=self.paths[i]) for i in range(self.n_taxa)] +
                      [Node(j) for j in range(self.n_taxa, self.n_nodes)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.clusters = self.nodes[:self.n_taxa]  # prende solo le foglie

        self.root = self.nodes[self.n_taxa]
        self.nj_torch()

    def nj_torch(self):
        # Prepara la matrice di distanza
        d = torch.tensor(self.d, device=self.device, dtype=torch.double)
        d = torch.max(d) + 1 - d
        # d = 1 - (d - torch.min(d)) / (torch.max(d) - torch.min(d))
        d.fill_diagonal_(0)

        # Setup per NJ
        max_val = d.max() * d.shape[0]
        adj_idxs = [i for i in range(self.n_taxa)]
        int_node = self.n_taxa + 1

        # finché restano >3 taxa
        while d.shape[0] > 3:
            # Calcola la Q-matrix e sceglie la coppia da unire
            s = d.sum(dim=-1).unsqueeze(0)

            q = (d.shape[0] - 2) * d - s - s.T
            q.fill_diagonal_(max_val)
            idxs = torch.argmin(q).item()
            minI, minJ = idxs // q.shape[0], idxs % q.shape[0]

            # Collega i nodi scelti e aggiorna la corrispondenza
            self.link(minI, minJ, int_node, adj_idxs)

            adj_idxs.pop(max(minI, minJ))
            adj_idxs[min(minI, minJ)] = int_node
            int_node += 1

            # Aggiorna la matrice delle distanze
            new_dist = torch.zeros(d.shape[0] - 1, device=self.device, dtype=torch.double)
            new_dist[: max(minI, minJ)] = (d[minI, : max(minI, minJ)] + d[minJ, : max(minI, minJ)] - d[minI, minJ]) / 2
            new_dist[max(minI, minJ):] = (d[minI, max(minI, minJ) + 1:] + d[minJ, max(minI, minJ) + 1:] - d[
                minI, minJ]) / 2

            d = torch.cat([d[:, : max(minI, minJ)], d[:, max(minI, minJ) + 1:]], dim=1)
            d = torch.cat([d[: max(minI, minJ), :], d[max(minI, minJ) + 1:, :]], dim=0)
            d[min(minI, minJ), :] = new_dist
            d[:, min(minI, minJ)] = new_dist

        # Chiusura con 3 taxa rimasti
        for i in range(3):
            self.nodes[adj_idxs[i]].links.append(self.nodes[self.n_taxa])
        self.nodes[self.n_taxa].links += [self.nodes[adj_idxs[i]] for i in range(3)]
        if sum(self.root.links[i].is_leaf for i in range(3)) == 2:
            self.root.is_cherry = True
            self.cherries.append(self.root)

        # self.draw() ------------- STAMPA L'ALBERO

    def link(self, minI, minJ, int_node, adj_idxs):
        self.nodes[adj_idxs[minI]].links.append(self.nodes[int_node])
        self.nodes[adj_idxs[minJ]].links.append(self.nodes[int_node])
        self.nodes[int_node].links += [self.nodes[adj_idxs[minI]], self.nodes[adj_idxs[minJ]]]
        if self.nodes[adj_idxs[minI]].is_leaf and self.nodes[adj_idxs[minJ]].is_leaf:
            self.nodes[int_node].is_cherry = True
            self.cherries.append(self.nodes[int_node])

    def draw(self):
        g = nx.Graph()
        g.add_node(self.root.id)
        for link in self.root.links:
            self.add_node(self.root, link, g)

        nx.draw(g, with_labels=True, node_size=10)
        plt.show()

    def add_node(self, parent, node, g):
        g.add_node(node.id)
        g.add_edge(parent.id, node.id)
        for link in node.links:
            if link.id != parent.id:
                self.add_node(node, link, g)
