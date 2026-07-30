import math
from collections import defaultdict
import networkx as nx
from mcpa.elements.agent import Agent
from mcpa.elements.path import Path
import utils.parallel as parallel

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
    def compute_k_shortest_paths(self, G, k, sim_method) -> None:
        gen = nx.shortest_simple_paths(G, self.src, self.dst)
        self.k_shortest_paths = [Path(next(gen)) for _ in range(k)]
        self.T = len(self.k_shortest_paths[-1].visits)
        self.delay_shortest_paths(self.T)
        weighting, beta = sim_method
        self._build_visits_count(weighting, beta)


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




    def _build_visits_count(self, weighting: str, beta: float | None):
        paths = self.all_paths

        min_len = len(self.k_shortest_paths[0].visits)

        raw_weights = []
        for path in paths:
            c = len(path.visits) - min_len

            if weighting == "uniform":
                w = 1.0
            elif weighting == "inverse":
                w = 1.0 / (c + 1.0)
            elif weighting == "exponential":
                w = math.exp(-beta * c)
            else:
                raise ValueError(f"'{weighting}' NOT supported")

            raw_weights.append(w)

        total_weight = sum(raw_weights)
        norm_weights = [w / total_weight for w in raw_weights]

        sig = defaultdict(float)
        for idx, path in enumerate(paths):
            enc = path.encoded
            w_norm = norm_weights[idx]

            for t, node_id in enumerate(enc):
                sig[(t, int(node_id))] += w_norm

        self.visit_counts = sig


    @staticmethod
    def compute_similarity(od1, od2) -> float:
        vc1 = od1.visit_counts
        vc2 = od2.visit_counts

        if len(vc1) > len(vc2):
            vc1, vc2 = vc2, vc1

        rw = parallel._RESOURCE_WEIGHTS or {}

        sim = 0.0
        for r, c1 in vc1.items():
            c2 = vc2.get(r)
            if c2:
                w_r = rw.get(r, 1.0)
                sim += c1 * c2 * w_r

        min_demand = min(len(od1.agents), len(od2.agents))

        if len(od1.all_paths) > 0 and len(od2.all_paths) > 0:
            sim = min_demand * sim
        else:
            sim = 0.0
        return sim

