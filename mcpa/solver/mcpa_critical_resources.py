from collections import defaultdict
import networkx as nx
import heapq

from mcpa.elements.agent import Agent
from general.general_critical_resources import General_Critical_Resources
from mcpa.elements.pair import OD_Pair


class MCPA_Critical_Resources(General_Critical_Resources):
    def __init__(
        self,
        G: nx.Graph,
        od_pairs: list[OD_Pair]
    ):
        self.G = G
        super().__init__(od_pairs)

        self.critical_od_pairs = set()


    def _compute_residuals(self):
        self.capacities = {v: self.G.nodes[v]["capacity"] for v in self.G.nodes}
        self.agents_per_resource = defaultdict(set)
        self.agents = [a for od in self.elements for a in od.agents]
        for a in self.agents:
            for t, v in enumerate(a.path.visits):
                self.agents_per_resource[(v, t)].add(a)

        return {(v, t): (self.capacities[v] - len(agents)) for (v, t), agents in self.agents_per_resource.items()}


    def _compute_relevant_resources(self):
        self.od_by_agent = {}
        self.paths_by_od = {}
        self.resources_by_path = {}

        for od in self.elements:
            all_paths_visits = [tuple(p.visits) for p in od.all_paths]
            self.paths_by_od[od] = all_paths_visits
            for path_visits in all_paths_visits:
                self.resources_by_path[path_visits] = [(path_visits[t], t) for t in range(len(path_visits))]

            for a in od.agents:
                self.od_by_agent[a] = od

        if not self.violated_resources:
            self.violated_od_pairs = set()
            return None

        self.violated_od_pairs = {
            self.od_by_agent[a]
            for vr in self.violated_resources
            for a in self.agents_per_resource.get(vr, ())
        }

        rel = {
            r
            for od in self.violated_od_pairs
            for path_visits in self.paths_by_od[od]
            for r in self.resources_by_path[path_visits]
        }
        return rel


    def _compute_score(self, candidate: Agent):
        a_visits = tuple(candidate.path.visits)
        alternatives = (visited_nodes for visited_nodes in self.paths_by_od[self.od_by_agent[candidate]] if
                        visited_nodes != a_visits)

        score = float("-inf")
        for alternative in alternatives:
            _min = float("inf")
            for (v, t) in self.resources_by_path[alternative]:
                occ = len(self.agents_per_resource.get((v, t), ()))
                res = self.residuals.get((v, t), self.capacities[v] - occ)
                if res < _min:
                    _min = res
                    if _min <= score:
                        break
            if _min > score:
                score = _min
        return score


    def _get_candidates(self, resource):
        return self.agents_per_resource.get(resource, ())

    def _recompute(self, agent: Agent):
        for t, v in enumerate(agent.path.visits):
            key = (v, t)
            self.agents_per_resource[key].discard(agent)
            if len(self.agents_per_resource[key]) == 0:
                self.agents_per_resource.pop(key)

            new_residual = self.residuals[key] + 1
            self.residuals[key] = new_residual

            if self.relevant_resources is not None and key not in self.relevant_resources:
                self.critical_resources.discard(key)
            else:
                if new_residual < self.current_tol:
                    heapq.heappush(self._heap, (new_residual, key))
                else:
                    self.critical_resources.discard(key)

            if new_residual == self.capacities[v]:
                self.residuals.pop(key, None)

        self.removed_items.add(agent)


    def _update_residuals(self):
        self.residuals = {(v, t): (self.capacities[v] - len(agents)) for (v, t), agents in self.agents_per_resource.items()}

    def _finalize_unassign(self):
        fixed = set(self.agents) - self.removed_items
        used_caps = defaultdict(int)
        for agent in fixed:
            for t, v in enumerate(agent.path.visits):
                used_caps[v, t] += 1
        self.left_caps = {(v, t): self.G.nodes[v]["capacity"] - used_cap for (v, t), used_cap in used_caps.items()}
        self.critical_od_pairs = set(od_pair for od_pair in self.elements if any(agent in self.removed_items for agent in od_pair.agents))
