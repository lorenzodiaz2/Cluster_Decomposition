from collections import defaultdict
from typing import List
import networkx as nx
from elements.agent import Agent
from elements.path import Path

class OD_Pair:
    def __init__(self, pair_id, src, dst, agents):
        self.id = pair_id
        self.src = src
        self.dst = dst
        self.k_shortest_paths: List[Path] = []
        self.delayed_shortest_paths: dict[int, list[Path]] = defaultdict(list)
        self.all_paths: List[Path] = []
        self.agents: List[Agent] = agents
        self.T = 0

    def __str__(self) -> str:
        return f"id = {self.id}    {self.src} , {self.dst}    ->    {len(self.agents)} agents,    {len(self.k_shortest_paths)} shortest paths"

    # todo gestire il caso in cui non ci sono k shortest path per una coppia
    def compute_k_shortest_paths(self, G, k) -> None:
        gen = nx.shortest_simple_paths(G, self.src, self.dst)
        self.k_shortest_paths = [Path(next(gen)) for _ in range(k)]
        self.T = len(self.k_shortest_paths[-1].visits) + 3
        self.delay_shortest_paths(self.T)

    def compute_similarity(
        self,
        other,
        all_paths_flag
    ) -> int:
        other_k_paths = other.k_shortest_paths

        if all_paths_flag:
            similarity = sum(path1.compare(path2) for path1 in self.all_paths for path2 in other.all_paths)
        else:
            similarity = sum(path1.compare(path2) for path1 in self.k_shortest_paths for path2 in other_k_paths) # * len(self.agents) * len(other.agents)

        return similarity

    def delay_shortest_paths(self, T: int) -> None:
        for idx, base_path in enumerate(self.k_shortest_paths):
            L = len(base_path.visits) - 1
            tau_max = T - L
            if tau_max <= 0:
                continue

            seen = {tuple(p.visits) for p in self.delayed_shortest_paths[idx]}

            for tau in range(1, tau_max + 1):
                visits_ext = [self.src] * tau + list(base_path.visits)
                if tuple(visits_ext) in seen:
                    continue
                self.delayed_shortest_paths[idx].append(Path(visits_ext))
        self.all_paths = self.get_all_paths()

    def get_all_paths(self):
        all_paths = list(self.k_shortest_paths)
        for paths_list in self.delayed_shortest_paths.values():
            all_paths.extend(paths_list)
        return all_paths