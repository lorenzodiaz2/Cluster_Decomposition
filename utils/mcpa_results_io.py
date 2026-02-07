import json
from typing import Any

from pandas import DataFrame
from scipy.odr import Model

from mcpa.elements.mcpa_environment import MCPA_Environment
from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from general.general_solver import General_Solver


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
                unassigned_agents = _safe_attr(critical_resources, "unassigned_agents_per_tol", None)
                unassigning_times = _safe_attr(critical_resources, "unassigning_times", None)
                total_time_clusters_post += sum(unassigning_times or [])

                model_times_final = _safe_attr(final_solver, "model_times", None)
                resolution_times_final = _safe_attr(final_solver, "resolution_times", None)
                status_final = _safe_attr(final_solver, "status", None)

                removed = set(_safe_attr(critical_resources, "removed_agents", None) or [])
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



