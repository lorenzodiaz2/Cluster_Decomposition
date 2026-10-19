from clustering.nj.nj_torch import NjTree, Node


class TreePartition(NjTree):

    def __init__(self, similarity_matrix, weights, max_cluster_size):
        super().__init__(similarity_matrix, weights, max_cluster_size)


    def get_min_cherry(self):
        return min(self.cherries, key=lambda ch: ch.compute_mass())

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
            mass = cherry.compute_mass()
            if mass <= self.max_cluster_size:
                cherry.merge(mass)
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
        self.clusters = sorted(self.clusters, key=lambda x: x.mass, reverse=True)




    def last_step(self, root: Node):
        nodes = sorted(root.adj_nodes, key=lambda node: node.mass)
        if len(nodes) == 3:
            mass = nodes[0].mass + nodes[1].mass

            if mass <= self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(nodes[1])
                root.mass = mass
                root.cluster += nodes[0].cluster + nodes[1].cluster
                self.clusters.append(root)

                if len(nodes) > 2:
                    mass = root.mass + nodes[2].mass
                    if mass < self.max_cluster_size:
                        root.mass = mass
                        root.cluster += nodes[2].cluster
                        self.clusters.remove(nodes[2])  ############################
        else:
            mass = root.mass + nodes[0].mass
            if mass < self.max_cluster_size:
                self.clusters.remove(nodes[0])
                self.clusters.remove(root)
                root.mass = mass
                root.cluster += nodes[0].cluster
                self.clusters.append(root)
