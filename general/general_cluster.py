import random
from abc import ABC, abstractmethod

from matplotlib.colors import to_hex

# todo potrebbe non servire questa classe

class General_Cluster(ABC):
    def __init__(self, cluster_id, elements):
        self.id = cluster_id
        self.elements = elements
        self.color = to_hex((random.random(), random.random(), random.random()))


    @abstractmethod
    def merge(self, other):
        raise NotImplementedError