import time
from collections import defaultdict
from typing import List
import networkx as nx
import heapq

from elements.pair import OD_Pair
from elements.agent import Agent


class Critical_Resources:
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: List[OD_Pair],
        tol: int = 0
    ):
        start = time.time()
        self.G = G
        self.od_pairs = od_pairs
        self.tol = tol

        self.cap = {v: G.nodes[v]["capacity"] for v in G.nodes}
        self.agents_per_resource = defaultdict(set)
        self.agents = [a for od in od_pairs for a in od.agents]
        for a in self.agents:
            for t, v in enumerate(a.path.visits):
                self.agents_per_resource[(v, t)].add(a)

        self.residuals = {key: (self.cap[key[0]] - len(agents)) for key, agents in self.agents_per_resource.items()}
        self.critical_resources = {key for key, res in self.residuals.items() if res < self.tol}
        self.critical_agents = {a for (v, t) in self.critical_resources for a in self.agents_per_resource[(v, t)]}

        self.od_by_agent = {}  # od_by_agent[a] -> od
        self.paths_by_od = {}  # paths_by_od[od] -> [visited_nodes di tutti i paths] però le salvo come tuple
        self.path_keys = {}    # tuple(visited_nodes) -> list[(v,t)]
        for od in od_pairs:
            visited_nodes_list = [tuple(p.visits) for p in od.all_paths]
            self.paths_by_od[od] = visited_nodes_list
            for visited_nodes in visited_nodes_list: # per ogni path
                self.path_keys[visited_nodes] = [(visited_nodes[t], t) for t in range(len(visited_nodes))]
            for a in od.agents:
                self.od_by_agent[a] = od

        self.scores: dict[Agent, float] = {}
        self.removed_agents = []
        self.critical_pairs = {}
        self.is_initially_feasible = all(res >= 0 for res in self.residuals.values())

        self._heap = []
        self._build_heap()

        self.creation_time = time.time() - start
        self.unassigning_time = None



    def assign_score(self, agents):
        self.scores.clear()
        for a in agents:
            a_visits = tuple(a.path.visits)
            alternatives = (visited_nodes for visited_nodes in self.paths_by_od[self.od_by_agent[a]] if visited_nodes != a_visits)

            score = float("-inf")
            for alternative in alternatives:
                minimum = float("inf")
                for (v, t) in self.path_keys[alternative]:
                    occ = len(self.agents_per_resource.get((v, t), ()))
                    r = self.residuals.get((v, t), self.cap[v] - occ)
                    if r < minimum:
                        minimum = r
                        if minimum <= score:
                            break
                if minimum > score:
                    score = minimum
            self.scores[a] = score


    def _build_heap(self):
        self._heap = [(self.residuals[k], k) for k in self.critical_resources]
        heapq.heapify(self._heap)


    def _pop_worst(self):
        while self._heap:
            res, k = heapq.heappop(self._heap)
            if k in self.critical_resources and self.residuals.get(k, float('inf')) == res:
                return k
        return None


    def unassign_agents(self):
        start = time.time()
        while self.critical_resources:
            worst_v, worst_t = self._pop_worst()
            candidate_agents = self.agents_per_resource.get((worst_v, worst_t), ())
            if not candidate_agents:
                self.critical_resources.discard((worst_v, worst_t))
                continue

            self.assign_score(candidate_agents)
            self.recompute(max(self.scores, key=self.scores.get))
        self.unassigning_time = time.time() - start


    def recompute(self, agent):
        for t, v in enumerate(agent.path.visits):
            key = (v, t)
            self.agents_per_resource[key].discard(agent)
            if len(self.agents_per_resource[key]) == 0:
                self.agents_per_resource.pop(key)

            new_residual = self.residuals[key] + 1
            self.residuals[key] = new_residual

            if new_residual < self.tol:
                heapq.heappush(self._heap, (new_residual, key))
            else:
                self.critical_resources.discard(key)

            if new_residual == self.cap[v]:
                self.residuals.pop(key, None)

        self.critical_agents.discard(agent)
        self.removed_agents.append(agent)


    def increment_tol(self, delta: int = 1) -> None:
        self.tol += delta
        self.residuals = {key: (self.cap[key[0]] - len(agents)) for key, agents in self.agents_per_resource.items()}
        self.critical_resources = {key for key, res in self.residuals.items() if res < self.tol}
        self.critical_agents = {a for (v, t) in self.critical_resources for a in self.agents_per_resource[(v, t)]}
        self._build_heap()

