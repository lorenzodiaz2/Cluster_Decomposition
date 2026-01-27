import networkx as nx
import torch
from matplotlib import pyplot as plt


class Node:
    def __init__(self, idx, is_leaf=False, is_cherry=False, n_agents=0):
        self.id = idx
        self.is_leaf = is_leaf
        self.is_cherry = is_cherry
        self.adj_nodes = []
        self.n_agents = n_agents
        self.cluster = [self.id] if is_leaf else []

    # Serve per sapere quanti agenti avrebbe un cluster se accettassi questo nodo come cluster
    def compute_n_agents(self):
        n_agents = 0
        for adj_node in self.adj_nodes:
            n_agents += adj_node.n_agents
        return n_agents

    def merge(self, n_agents):
        self.n_agents += n_agents
        for adj_node in self.adj_nodes:
            if adj_node.is_leaf:
                self.cluster += adj_node.cluster

        self.is_leaf = True
        self.is_cherry = False

    def get_parent(self):
        parent = None
        for adj_node in self.adj_nodes:
            if not adj_node.is_leaf:
                parent = adj_node
        return parent



class NjTree:
    def __init__(self, similarity_matrix, n_agents, max_cluster_size):
        self.d = similarity_matrix
        self.n_taxa = similarity_matrix.shape[0]
        self.max_cluster_size = max_cluster_size
        self.cherries = []
        self.n_nodes = 2 * self.n_taxa - 2
        self.n_agents = n_agents

        self.nodes = ([Node(i, is_leaf=True, n_agents=self.n_agents[i]) for i in range(self.n_taxa)] +
                      [Node(j) for j in range(self.n_taxa, self.n_nodes)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.clusters = self.nodes[:self.n_taxa]  # prende solo le foglie

        self.root = self.nodes[self.n_taxa]
        self.nj_torch()

    def nj_torch(self):
        # Prepara la matrice di distanza
        d = torch.tensor(self.d, device=self.device, dtype=torch.double)
        d = torch.max(d) + 1 - d
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
            self.nodes[adj_idxs[i]].adj_nodes.append(self.nodes[self.n_taxa])
        self.nodes[self.n_taxa].adj_nodes += [self.nodes[adj_idxs[i]] for i in range(3)]
        if sum(self.root.adj_nodes[i].is_leaf for i in range(3)) == 2:
            self.root.is_cherry = True
            self.cherries.append(self.root)

        # self.draw() ------------- STAMPA L'ALBERO

    def link(self, minI, minJ, int_node, adj_idxs):
        self.nodes[adj_idxs[minI]].adj_nodes.append(self.nodes[int_node])
        self.nodes[adj_idxs[minJ]].adj_nodes.append(self.nodes[int_node])
        self.nodes[int_node].adj_nodes += [self.nodes[adj_idxs[minI]], self.nodes[adj_idxs[minJ]]]
        if self.nodes[adj_idxs[minI]].is_leaf and self.nodes[adj_idxs[minJ]].is_leaf:
            self.nodes[int_node].is_cherry = True
            self.cherries.append(self.nodes[int_node])

    def draw(self):
        g = nx.Graph()
        g.add_node(self.root.id)
        for link in self.root.adj_nodes:
            self.add_node(self.root, link, g)

        nx.draw(g, with_labels=True, node_size=10)
        plt.show()

    def add_node(self, parent, node, g):
        g.add_node(node.id)
        g.add_edge(parent.id, node.id)
        for adj_node in node.adj_nodes:
            if adj_node.id != parent.id:
                self.add_node(node, adj_node, g)
