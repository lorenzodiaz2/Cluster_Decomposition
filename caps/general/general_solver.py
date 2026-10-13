import time
from functools import partial
import gurobipy as gp
from gurobipy import GRB, Model
from abc import ABC, abstractmethod


class Incumbent:
    def __init__(self):
        self.times = []
        self.solutions = []

def add_current_sol(model: Model, where, incumbent_obj):
    if where == GRB.Callback.MIPSOL:
        incumbent_obj.solutions.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        incumbent_obj.times.append(time.perf_counter() - model._start_time)


class General_Solver(ABC):
    def __init__(
        self,
        time_limit: int,
        verbose: bool
    ):
        self.m = None
        self.status = None
        self.time_limit = time_limit
        self.model_times = []
        self.resolution_times = []
        self.output_flag = 1 if verbose else 0
        self.gap = None
        self.incumbent = Incumbent()


    def solve(self):
        self._set_model()
        self._optimize_model()
        while self.status == "INFEASIBLE":
            changed = self._handle_infeasibility()
            if not changed:
                break
            self._set_model()
            self._optimize_model()
        self._assign_solutions()


    def _set_model(self):
        start = time.perf_counter()

        self.m = gp.Model("heuristic")
        self.m.Params.OutputFlag = self.output_flag

        self._set_variables()
        self._set_constraints()
        self._set_objective()

        self.model_times.append(time.perf_counter() - start)


    def _optimize_model(self):
        start = time.perf_counter()
        self.m.Params.TimeLimit = self.time_limit
        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.m._start_time = time.perf_counter()
        self.m.optimize(callback)
        self.resolution_times.append(time.perf_counter() - start)

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        self.status = status_map.get(self.m.Status, str(self.m.Status))
        self.gap = self.m.MIPGap if self.status != "INFEASIBLE" else None


    @abstractmethod
    def _set_variables(self):
        raise NotImplementedError

    @abstractmethod
    def _set_constraints(self):
        raise NotImplementedError

    @abstractmethod
    def _set_objective(self):
        raise NotImplementedError

    @abstractmethod
    def _handle_infeasibility(self):
        raise NotImplementedError

    @abstractmethod
    def _assign_solutions(self):
        raise NotImplementedError