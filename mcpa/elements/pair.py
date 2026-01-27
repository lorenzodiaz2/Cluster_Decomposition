from collections import defaultdict
import networkx as nx
from mcpa.elements.agent import Agent
from mcpa.elements.path import Path

class OD_Pair:
    def __init__(self, pair_id, src, dst, agents):
        self.id = pair_id
        self.src = src
        self.dst = dst
        self.k_shortest_paths: list[Path] = []
        self.delayed_shortest_paths: dict[int, list[Path]] = defaultdict(list)
        self.all_paths: list[Path] = []
        self.agents: list[Agent] = agents
        self.T = 0
        self.visit_counts = None

    def __str__(self) -> str:
        return f"id = {self.id}    {self.src} , {self.dst}    ->    {len(self.agents)} agents,    {len(self.k_shortest_paths)} shortest paths"

    # todo gestire il caso in cui non ci sono k shortest path per una coppia
    def compute_k_shortest_paths(self, G, k) -> None:
        gen = nx.shortest_simple_paths(G, self.src, self.dst)
        self.k_shortest_paths = [Path(next(gen)) for _ in range(k)]
        self.T = len(self.k_shortest_paths[-1].visits)
        self.delay_shortest_paths(self.T)
        self._build_visits_count()


    @staticmethod
    def compute_similarity(od1, od2) -> int:
        vc1 = od1.visit_counts
        vc2 = od2.visit_counts

        if len(vc1) > len(vc2):
            vc1, vc2 = vc2, vc1

        sim = 0
        for (t, v), c1 in vc1.items():
            c2 = vc2.get((t, v))
            if c2:
                sim += c1 * c2
        return sim

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
        self.all_paths = self._get_all_paths()

    def _get_all_paths(self):
        all_paths = list(self.k_shortest_paths)
        for paths_list in self.delayed_shortest_paths.values():
            all_paths.extend(paths_list)
        return all_paths


    def _build_visits_count(self):
        paths = self.all_paths

        # chiave = (t, node_id)  ->  value = conteggio
        sig = defaultdict(int)
        for path in paths:
            enc = path.encoded
            for t, node_id in enumerate(enc):
                sig[(t, int(node_id))] += 1

        self.visit_counts = sig
