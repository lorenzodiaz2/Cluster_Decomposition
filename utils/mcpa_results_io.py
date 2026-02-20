import math
import json
from typing import Any

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.odr import Model

from mcpa.elements.mcpa_environment import MCPA_Environment
from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from general.general_solver import General_Solver
from utils.read_instance import fmt_int, fmt_float, get_sum_from_array_string, get_last_from_array_string


# -----------------------------
# Helpers (robust + CSV friendly)
# -----------------------------
def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    try:
        return _jsonable(x.item())  # numpy scalar
    except Exception:
        return str(x)


def _dump(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    return json.dumps(_jsonable(x), ensure_ascii=False)


def _safe_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _safe_objVal(solver: General_Solver):
    m: Model = _safe_attr(solver, "m", None)
    if m is None:
        return None
    if getattr(m, "SolCount", 0) and getattr(m, "SolCount") > 0:
        return float(m.ObjVal)
    return None



def _safe_objBound(solver: General_Solver):
    m: Model = _safe_attr(solver, "m", None)
    if m is None:
        return None
    try:
        return float(m.ObjBound)
    except Exception:
        return None


def _sum_or_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) != len(values):
        return None
    return float(sum(vals))


def _extract_cluster_block(
    env: MCPA_Environment,
    UB_complete: float,
    critical_resources: MCPA_Critical_Resources | None,
    final_solver: MCPA_Heuristic_Solver | None,
) -> dict[str, Any]:

    # cluster stats
    n_clusters = len(env.clusters)
    n_agents_per_cluster = [c.n_agents for c in env.clusters]
    od_pairs_per_cluster = [[od.id for od in c.elements] for c in env.clusters]

    model_times_clusters = []
    resolution_times_clusters = []
    clusters_status = []
    UBs_clusters = []
    LBs_clusters = []

    for hs in env.clusters_solvers:
        model_times_clusters.append(_safe_attr(hs, "model_times", [None])[0] if _safe_attr(hs, "model_times", None) else None)
        resolution_times_clusters.append(_safe_attr(hs, "resolution_times", [None])[0] if _safe_attr(hs, "resolution_times", None) else None)
        clusters_status.append(_safe_attr(hs, "status", None))
        UBs_clusters.append(_safe_objVal(hs))
        LBs_clusters.append(_safe_objBound(hs))

    UB_clusters = _sum_or_none(UBs_clusters)
    LB_clusters = _sum_or_none(LBs_clusters)

    critical_resources_creation_times = None
    unassigned_agents = None
    unassigning_times = None

    model_times_final = None
    resolution_times_final = None
    status_final = None

    total_time_clusters_post = (env.matrix_time or 0.0) + (env.nj_time or 0.0)

    for hs in env.clusters_solvers:
        total_time_clusters_post += sum(_safe_attr(hs, "model_times", []) or [])
        total_time_clusters_post += sum(_safe_attr(hs, "resolution_times", []) or [])

    total_delay = sum(a.delay for a in env.agents)

    cluster_has_tl = any(s == "TIME_LIMIT" for s in (clusters_status or []))
    gap = None

    if cluster_has_tl:
        UB_final = None
        LB_final = None
    else:
        if critical_resources is not None:
            critical_resources_creation_times = _safe_attr(critical_resources, "creation_times", None)
            total_time_clusters_post += sum(critical_resources_creation_times or [])

            if final_solver is not None:
                unassigned_agents = _safe_attr(critical_resources, "unassigned_items_per_tol", None)
                unassigning_times = _safe_attr(critical_resources, "unassigning_times", None)
                total_time_clusters_post += sum(unassigning_times or [])

                model_times_final = _safe_attr(final_solver, "model_times", None)
                resolution_times_final = _safe_attr(final_solver, "resolution_times", None)
                status_final = _safe_attr(final_solver, "status", None)

                removed = set(_safe_attr(critical_resources, "removed_items", None) or [])
                fixed_delay = sum(a.delay for a in env.agents if a not in removed)

                ub_removed = _safe_objVal(final_solver)
                lb_removed = _safe_objBound(final_solver)

                UB_final = (fixed_delay + ub_removed) if ub_removed is not None else None
                LB_final = (fixed_delay + lb_removed) if lb_removed is not None else None

                total_time_clusters_post += sum(model_times_final or [])
                total_time_clusters_post += sum(resolution_times_final or [])
                gap = 100 * (UB_final - UB_complete) / UB_complete
            else:
                gap = 0. if UB_complete == total_delay else 100 * (total_delay - UB_complete) / UB_complete
                UB_final = total_delay
                LB_final = total_delay
        else:
            UB_final = total_delay
            LB_final = total_delay


    out = {
        f"n clusters": n_clusters,
        f"similarity index": _safe_attr(env, "similarity_index", None),
        f"cluster similarity indexes": _dump(_safe_attr(env, "cluster_similarity_indexes", None)),
        f"cluster congestion indexes": _dump(_safe_attr(env, "cluster_congestion_indexes", None)),
        f"cluster congestion indexes absolute": _dump(_safe_attr(env, "cluster_congestion_indexes_absolute", None)),
        f"cluster congestion ratio max": _dump(_safe_attr(env, "cluster_congestion_ratio_max", None)),

        f"global congestion absolute": _dump(_safe_attr(env, "global_congestion_index_absolute", None)),
        f"cross congestion absolute": _dump(_safe_attr(env, "cross_congestion_index_absolute", None)),
        f"global congestion ratio max": _dump(_safe_attr(env, "global_congestion_ratio_max", None)),
        f"cross congestion rate": _dump(_safe_attr(env, "cross_congestion_rate", None)),
        f"cross congestion share": _dump(_safe_attr(env, "cross_congestion_share", None)),

        f"similarity matrix time": _safe_attr(env, "matrix_time", None),
        f"nj time": _safe_attr(env, "nj_time", None),

        f"n agents per cluster": _dump(n_agents_per_cluster),
        f"od pairs per cluster": _dump(od_pairs_per_cluster),

        f"model times clusters": _dump(model_times_clusters),
        f"resolution times clusters": _dump(resolution_times_clusters),
        f"clusters status": _dump(clusters_status),
        f"UBs clusters": _dump(UBs_clusters),
        f"LBs clusters": _dump(LBs_clusters),

        f"critical resources creation times": _dump(critical_resources_creation_times),
        f"unassigned agents": _dump(unassigned_agents),
        f"unassigning agents times": _dump(unassigning_times),

        f"model times final": _dump(model_times_final),
        f"resolution times final": _dump(resolution_times_final),
        f"status final": status_final,

        f"UB clusters": UB_clusters,
        f"LB clusters": LB_clusters,
        f"UB final": UB_final,
        f"LB final": LB_final,
        f"gap": gap,
        f"total time clusters + post": float(total_time_clusters_post),
    }
    print(f"t = {round(total_time_clusters_post, 2)}  LB = {LB_final}   UB = {UB_final}    ({n_clusters})")
    return out




def save_mcpa_results(
    env: MCPA_Environment,
    critical_resources: MCPA_Critical_Resources | None,
    final_solver: MCPA_Heuristic_Solver | None,
    complete_solver: MCPA_Heuristic_Solver,
    seed: int,
    df: DataFrame,
):
    # -------------------------
    # Parameters / instance info
    # -------------------------
    grid_side = env.grid_side
    n_quadrants = env.n_quadrants
    n_pairs_per_quadrant = env.n_elements_per_quadrant
    n_agents = len(env.agents)
    max_cluster_size = env.max_cluster_size
    offset = env.offset
    k = env.k
    env_time = env.set_time

    # -------------------------
    # Complete solution block
    # -------------------------
    model_times_complete = _safe_attr(complete_solver, "model_times", None)
    resolution_times_complete = _safe_attr(complete_solver, "resolution_times", None)
    status_complete = _safe_attr(complete_solver, "status", None)

    UB_complete = _safe_objVal(complete_solver)
    LB_complete = _safe_objBound(complete_solver)
    gap_complete = _safe_attr(complete_solver, "gap", None)

    incumbent = _safe_attr(complete_solver, "incumbent", None)
    incumbent_times = _safe_attr(incumbent, "times", []) if incumbent is not None else []
    incumbent_solutions = _safe_attr(incumbent, "solutions", []) if incumbent is not None else []

    total_time_complete = 0.0
    if model_times_complete is not None:
        total_time_complete += sum(model_times_complete)
    if resolution_times_complete is not None:
        total_time_complete += sum(resolution_times_complete)
    print(f"t = {round(total_time_complete, 2)}  LB = {LB_complete}  UB = {UB_complete}", end="   -   ")

    # -------------------------
    # Cluster blocks (1 and 2)
    # -------------------------
    block = _extract_cluster_block(env, UB_complete, critical_resources, final_solver)

    # -------------------------
    # Build row dict following your DF schema
    # -------------------------
    row = {
        # PARAMETRI DI MODELLO
        "grid side": int(grid_side),
        "n quadrants": int(n_quadrants),
        "n pairs per quadrant": int(n_pairs_per_quadrant),
        "n agents": int(n_agents),
        "max cluster size": int(max_cluster_size),
        "offset": int(offset),
        "k": int(k),
        "seed": int(seed),
        "time limit": complete_solver.time_limit,
        "env time": float(env_time) if env_time is not None else None,

        # VALORI SOLUZIONE COMPLETA
        "model times complete": _dump(model_times_complete),
        "resolution times complete": _dump(resolution_times_complete),
        "status complete": status_complete,
        "UB complete": UB_complete,
        "LB complete": LB_complete,
        "gap complete": float(gap_complete) if gap_complete is not None else None,
        "incumbent times": _dump(incumbent_times),
        "incumbent solutions": _dump(incumbent_solutions),
        "total time complete": float(total_time_complete),
    }

    row.update(block)

    df.loc[len(df)] = row


def save_mcpa_tex_tables():
    column_formatters = {
        "1": fmt_int,
        "2": fmt_int,
        "3": fmt_float,
        "4": fmt_int,
        "5": fmt_float,
        "6": fmt_int,
        "7": fmt_float,
        "8": fmt_float,
        "9": fmt_float,
        "10": fmt_int,
        "11": fmt_float,
        "12": fmt_float
    }

    df = pd.read_csv("results/mcpa/mcpa_results.csv")
    df['unassigning agents times'] = df['unassigning agents times'].astype(str)
    df['model times final'] = df['model times final'].astype(str)
    df['resolution times final'] = df['resolution times final'].astype(str)
    df['critical resources creation times'] = df['critical resources creation times'].astype(str)
    df['unassigned agents'] = df['unassigned agents'].astype(str)



    for offset in [-1, 0, 1, 2, 5]:
        df_out = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        for n_clients_per_quadrant in [140, 145, 150, 155, 160]:
            for n_quadrants in [2, 3, 4, 6, 7, 9]:
                n_instances = 0
                complete_times = []
                heuristic_times = []
                gaps = []
                n_time_limit_complete = 0
                n_time_limit_heuristic = 0
                speeds_up = []
                repair_times = []
                unassigned_agents = []
                n_clusters = []
                cluster_creation_times = []

                for _, row in df.iterrows():
                    _grid_side = int(row["grid side"])
                    _n_quadrants = int(row["n quadrants"])
                    _n_clients_per_quadrant = int(row["n pairs per quadrant"])
                    _n_agents = float(row["n agents"])
                    _max_cluster_size = int(row["max cluster size"])
                    _offset = int(row["offset"])
                    _k = int(row["k"])
                    _seed = int(row["seed"])
                    _status_complete = row["status complete"]
                    _clusters_status = row["clusters status"]
                    _complete_time = float(row["total time complete"])
                    _heuristic_time = float(row["total time clusters + post"])
                    _status_final = row["status final"]
                    _n_clusters = int(row["n clusters"])
                    _cluster_creation_time = 100 * (float(row["similarity matrix time"]) + float(row["nj time"])) / _heuristic_time

                    _repair_time = 0

                    if row["critical resources creation times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["critical resources creation times"])

                    if row["unassigning agents times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["unassigning agents times"])

                    if row["model times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["model times final"])

                    if row["resolution times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["resolution times final"])

                    _unassigned_agents = 0
                    if row["unassigned agents"] != "nan":
                        _unassigned_agents += get_last_from_array_string(row["unassigned agents"])

                    consider_gap = True
                    _gap = float(row["gap"])

                    if n_clients_per_quadrant != _n_clients_per_quadrant or n_quadrants != _n_quadrants or offset != _offset:
                        continue

                    if _status_complete == "TIME_LIMIT":
                        n_time_limit_complete += 1

                    if "TIME_LIMIT" in _clusters_status:
                        n_time_limit_heuristic += 1
                        consider_gap = False

                    if _status_final == "TIME_LIMIT":
                        if "TIME_LIMIT" not in _clusters_status:
                            n_time_limit_heuristic += 1

                    n_instances += 1
                    complete_times.append(_complete_time)
                    heuristic_times.append(_heuristic_time)
                    if consider_gap:
                        gaps.append(_gap)
                    speeds_up.append(_complete_time / _heuristic_time)
                    repair_times.append(100 * _repair_time / _heuristic_time)
                    unassigned_agents.append(100 * _unassigned_agents / _n_agents)
                    n_clusters.append(_n_clusters)
                    cluster_creation_times.append(_cluster_creation_time)

                if n_instances < 5:
                    continue

                if n_time_limit_complete > 5:
                    n_time_limit_complete = 5
                if n_time_limit_heuristic > 5:
                    n_time_limit_heuristic = 5
                median_complete_times = round(np.median(complete_times), 2)
                median_heuristic_times = round(np.median(heuristic_times), 2)
                median_gaps = round(np.median(gaps), 2) if len(gaps) > 0 else None
                std_dev_gap = round(math.sqrt(np.std(gaps)), 2) if len(gaps) > 0 else None
                median_speed_up = round(np.median(speeds_up), 2) if len(speeds_up) > 0 else None
                median_repair_times = round(np.median(repair_times), 2) if len(repair_times) > 0 else 0
                median_unassigned_agents = round(np.median(unassigned_agents), 2) if len(unassigned_agents) > 0 else 0
                median_n_clusters = np.median(n_clusters) if len(n_clusters) > 0 else 0
                median_clusters_creation_time = np.median(cluster_creation_times) if len(cluster_creation_times) > 0 else 0

                df_out.loc[len(df_out)] = [n_clients_per_quadrant, n_quadrants, median_complete_times, n_time_limit_complete, median_heuristic_times, n_time_limit_heuristic, median_clusters_creation_time, median_repair_times, median_unassigned_agents, median_n_clusters, median_gaps, median_speed_up]

        df_out.to_latex(
            f"results/mcpa/tex/summary_{offset}.tex",
            index=False,
            formatters=column_formatters
        )

