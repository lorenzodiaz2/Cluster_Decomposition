from general.general_cluster import General_Cluster


class MCPA_Cluster(General_Cluster):
    def __init__(self, cluster_id, od_pairs):
        super().__init__(cluster_id, od_pairs)
        self.n_agents = sum(len(od_pair.agents) for od_pair in self.elements)


    def __str__(self) -> str:
        return f"Cluster {self.id}   number of agents = {self.n_agents}     pairs = {[pair.id for pair in self.elements]}"


    def merge(self, other) -> None:
        self.elements.extend(other.elements)
        self.n_agents = sum(len(od_pair.agents) for od_pair in self.elements)