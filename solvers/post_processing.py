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
        start = time.perf_counter()
        self.G = G
        self.od_pairs = od_pairs
        self.current_tol = tol if tol >= 0 else 0

        self.cap = {v: G.nodes[v]["capacity"] for v in G.nodes}
        self.agents_per_resource = defaultdict(set)
        self.agents = [a for od in self.od_pairs for a in od.agents]
        for a in self.agents:
            for t, v in enumerate(a.path.visits):
                self.agents_per_resource[(v, t)].add(a)

        self.residuals = {(v, t): (self.cap[v] - len(agents)) for (v, t), agents in self.agents_per_resource.items()}
        self.critical_resources = {(v, t) for (v, t), res in self.residuals.items() if res < self.current_tol}

        self.od_by_agent = {}  # od_by_agent[a] -> od
        self.paths_by_od = {}  # paths_by_od[od] -> [path_visits di tutti i paths] però le salvo come tuple
        self.resources_by_path = {}    # tuple(path_visits) -> list[(v,t)]
        for od in od_pairs:
            all_paths_visits = [tuple(p.visits) for p in od.all_paths]
            self.paths_by_od[od] = all_paths_visits
            for path_visits in all_paths_visits:
                self.resources_by_path[path_visits] = [(path_visits[t], t) for t in range(len(path_visits))]
            for a in od.agents:
                self.od_by_agent[a] = od

        self.scores: dict[Agent, float] = {}
        self.removed_agents = set()
        self.is_initially_feasible = all(res >= 0 for res in self.residuals.values())
        self.critical_od_pairs = set()
        self.left_caps = None

        self._heap = []
        self._build_heap()

        # queste variabili mi servono solo per il salvataggio dei risultati, non per l'algoritmo
        self.creation_times = []
        self.creation_times.append(time.perf_counter() - start)
        self.unassigning_times = []
        self.starting_tol = tol if tol >= 0 else 0
        self.unassigned_agents_per_tol = []
        self.critical_resources_per_tol = []
        self.critical_resources_per_tol.append(len(self.critical_resources))


    def _assign_score(self, agents):
        self.scores.clear()
        for a in agents:
            a_visits = tuple(a.path.visits)
            alternatives = (visited_nodes for visited_nodes in self.paths_by_od[self.od_by_agent[a]] if visited_nodes != a_visits)

            score = float("-inf")
            for alternative in alternatives:
                _min = float("inf")
                for (v, t) in self.resources_by_path[alternative]:
                    occ = len(self.agents_per_resource.get((v, t), ()))
                    res = self.residuals.get((v, t), self.cap[v] - occ)
                    if res < _min:
                        _min = res
                        if _min <= score:
                            break
                if _min > score:
                    score = _min
            self.scores[a] = score


    def _build_heap(self):
        self._heap = [(self.residuals[v, t], (v, t)) for (v, t) in self.critical_resources]
        heapq.heapify(self._heap)


    def _pop_worst(self):
        while self._heap:
            res, (v, t) = heapq.heappop(self._heap)
            if (v, t) in self.critical_resources and self.residuals.get((v, t), float('inf')) == res:
                return v, t
        return None


    def unassign_agents(self):
        start = time.perf_counter()
        while self.critical_resources:
            worst_v, worst_t = self._pop_worst()
            candidate_agents = self.agents_per_resource.get((worst_v, worst_t), ())
            # todo in teoria non dovrebbe mai entrare in questo if
            if not candidate_agents:
                self.critical_resources.discard((worst_v, worst_t))
                continue

            self._assign_score(candidate_agents)
            self._recompute(max(self.scores, key=self.scores.get))

        fixed = set(self.agents) - self.removed_agents
        used_caps = defaultdict(int)
        for agent in fixed:
            for t, v in enumerate(agent.path.visits):
                used_caps[v, t] += 1
        self.left_caps = {(v, t): self.G.nodes[v]["capacity"] - used_cap for (v, t), used_cap in used_caps.items()}
        self.critical_od_pairs = set(od_pair for od_pair in self.od_pairs if any(agent in self.removed_agents for agent in od_pair.agents))

        self.unassigning_times.append(time.perf_counter() - start)
        self.unassigned_agents_per_tol.append(len(self.removed_agents))


    def _recompute(self, agent):
        for t, v in enumerate(agent.path.visits):
            key = (v, t)
            self.agents_per_resource[key].discard(agent)
            if len(self.agents_per_resource[key]) == 0:
                self.agents_per_resource.pop(key)

            new_residual = self.residuals[key] + 1
            self.residuals[key] = new_residual

            if new_residual < self.current_tol:
                heapq.heappush(self._heap, (new_residual, key))
            else:
                self.critical_resources.discard(key)

            if new_residual == self.cap[v]:
                self.residuals.pop(key, None)

        self.removed_agents.add(agent)


    def increment_tol(self, delta: int = 1) -> None:
        start = time.perf_counter()
        self.current_tol += delta
        self.residuals = {(v, t): (self.cap[v] - len(agents)) for (v, t), agents in self.agents_per_resource.items()}
        self.critical_resources = {(v, t) for (v, t), res in self.residuals.items() if res < self.current_tol}
        self._build_heap()

        self.creation_times.append(time.perf_counter() - start)
        self.critical_resources_per_tol.append(len(self.critical_resources))
