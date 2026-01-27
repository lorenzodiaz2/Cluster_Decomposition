from general.general_cluster import General_Cluster


class CFL_Cluster(General_Cluster):
    def __init__(self, cluster_id, clients, facilities):
        super().__init__(cluster_id, clients)
        self.facilities = facilities


    def __str__(self) -> str:
        return f"Cluster {self.id}   n clients = {len(self.elements)}" #   clients = {[client.id for client in self.elements]}   facilities = {[f.id for f in self.facilities]}"


    def merge(self, other) -> None:
        self.elements.extend(other.elements)
        self.facilities.extend(other.facilities)

        new_el = sorted(list(set(self.elements)), key=lambda el: el.id)
        new_fac = sorted(list(set(self.facilities)), key=lambda fac: fac.id)
        self.elements = new_el
        self.facilities = new_fac
