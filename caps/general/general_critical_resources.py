import heapq
import time
from abc import ABC, abstractmethod


class General_Critical_Resources(ABC):
    def __init__(
        self,
        elements: list,
    ):
        start = time.perf_counter()
        self.elements = elements
        self.current_tol = 0
        self.residuals = self._compute_residuals()
        self.violated_resources = {r for r, res in self.residuals.items() if res < 0}
        self.critical_resources: set = {r for r, res in self.residuals.items() if res < self.current_tol}
        self.relevant_resources = self._compute_relevant_resources()
        if self.relevant_resources is not None:
            self.critical_resources &= self.relevant_resources

        self.is_initially_feasible = all(res >= 0 for res in self.residuals.values())
        self.scores = {}

        self.left_caps = None
        self.released_units = set()

        self._heap = []
        self._build_heap()


        # solo per logging
        self.creation_times = [time.perf_counter() - start]
        self.release_times = []
        self.starting_tol = 0
        self.released_units_per_tol = []
        self.critical_resources_per_tol = [len(self.critical_resources)]



    def release_units(self):
        start = time.perf_counter()
        while self.critical_resources:
            worst_resource = self._pop_worst()
            if worst_resource is None:
                break

            candidates = self._get_candidates(worst_resource)

            if not candidates:
                self.critical_resources.discard(worst_resource)
                continue

            self._assign_scores(candidates)

            if not self.scores:
                self.critical_resources.discard(worst_resource)
                continue

            self._recompute(max(self.scores, key=self.scores.get))

        self._finalize_release()
        self.release_times.append(time.perf_counter() - start)
        self.released_units_per_tol.append(len(self.released_units))


    def _assign_scores(self, candidates):
        self.scores.clear()
        for candidate in candidates:
            self.scores[candidate] = self._compute_score(candidate)


    def _build_heap(self):
        self._heap = [(self.residuals[resource], resource) for resource in self.critical_resources]
        heapq.heapify(self._heap)


    def _pop_worst(self):
        while self._heap:
            residual, resource = heapq.heappop(self._heap)
            if resource in self.critical_resources and self.residuals.get(resource, float('inf')) == residual:
                return resource
        return None

    def increment_tol(self):
        start = time.perf_counter()

        self.current_tol += 1
        self._update_residuals()

        self.violated_resources = {r for r, res in self.residuals.items() if res < 0}

        prev_rel = self.relevant_resources
        new_rel = self._compute_relevant_resources()

        if new_rel is None and prev_rel is not None:
            new_rel = prev_rel

        self.relevant_resources = new_rel

        base_critical = {r for r, res in self.residuals.items() if res < self.current_tol}
        self.critical_resources = (
            base_critical & self.relevant_resources
            if self.relevant_resources is not None
            else base_critical
        )

        self._build_heap()
        self.creation_times.append(time.perf_counter() - start)
        self.critical_resources_per_tol.append(len(self.critical_resources))

    @abstractmethod
    def _update_residuals(self):
        raise NotImplementedError

    @abstractmethod
    def _recompute(self, candidate):
        raise NotImplementedError

    @abstractmethod
    def _get_candidates(self, resource):
        raise NotImplementedError

    @abstractmethod
    def _compute_score(self, candidate):
        raise NotImplementedError

    @abstractmethod
    def _compute_residuals(self):
        raise NotImplementedError

    @abstractmethod
    def _finalize_release(self):
        raise NotImplementedError

    @abstractmethod
    def _compute_relevant_resources(self):
        raise NotImplementedError