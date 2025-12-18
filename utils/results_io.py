import json
from typing import Any

from pandas import DataFrame

from elements.environment import Environment
from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources


# -----------------------------
# Helpers (robust + CSV friendly)
# -----------------------------
def _jsonable(x: Any) -> Any:
    """Convert to something JSON-serializable (stable for CSV)."""
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        # stable order
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, dict):
        # stable keys order
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    # numpy scalars, gurobi objs, custom objs, etc.
    try:
        return _jsonable(x.item())  # numpy scalar
    except Exception:
        return str(x)


def _dump(x: Any) -> Any:
    """Dump containers to JSON string; leave scalars as-is."""
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    return json.dumps(_jsonable(x), ensure_ascii=False)


def _safe_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _safe_objval(solver: Heuristic_Solver):
    m = _safe_attr(solver, "m", None)
    if m is None:
        return None
    try:
        # ObjVal exists only if there is at least one solution
        if getattr(m, "SolCount", 0) and getattr(m, "SolCount") > 0:
            return float(m.ObjVal)
        return None
    except Exception:
        return None


def _safe_objbound(solver: Heuristic_Solver):
    m = _safe_attr(solver, "m", None)
    if m is None:
        return None
    try:
        return float(m.ObjBound)
    except Exception:
        return None


def _sum_or_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) != len(values):  # if any missing => return None (più onesto)
        return None
    return float(sum(vals))


def _extract_cluster_block(
    env: Environment,
    cluster_solvers: list[Heuristic_Solver],
    critical_resources: Critical_Resources | None,
    final_solver: Heuristic_Solver | None,
    suffix: str,  # " 1" or " 2"
) -> dict[str, Any]:

    # cluster stats
    n_clusters = len(env.clusters)
    n_agents_per_cluster = [c.n_agents for c in env.clusters]
    od_pairs_per_cluster = [[od.id for od in c.od_pairs] for c in env.clusters]

    # per-cluster solver stats
    model_times_clusters = []
    resolution_times_clusters = []
    clusters_status = []
    UBs_clusters = []
    LBs_clusters = []

    for hs in cluster_solvers:
        model_times_clusters.append(_safe_attr(hs, "model_times", [None])[0] if _safe_attr(hs, "model_times", None) else None)
        resolution_times_clusters.append(_safe_attr(hs, "resolution_times", [None])[0] if _safe_attr(hs, "resolution_times", None) else None)
        clusters_status.append(_safe_attr(hs, "status", None))
        UBs_clusters.append(_safe_objval(hs))
        LBs_clusters.append(_safe_objbound(hs))

    UB_clusters = _sum_or_none(UBs_clusters)
    LB_clusters = _sum_or_none(LBs_clusters)

    # post-processing / repair stats
    critical_resources_creation_times = None
    unassigned_agents = None
    unassigning_times = None

    model_times_final = None
    resolution_times_final = None
    status_final = None

    total_time_clusters_post = (env.matrix_time or 0.0) + (env.nj_time or 0.0)

    # add cluster solving times
    for hs in cluster_solvers:
        total_time_clusters_post += sum(_safe_attr(hs, "model_times", []) or [])
        total_time_clusters_post += sum(_safe_attr(hs, "resolution_times", []) or [])

    if critical_resources is not None:
        critical_resources_creation_times = _safe_attr(critical_resources, "creation_times", None)
        if critical_resources_creation_times is not None:
            total_time_clusters_post += sum(critical_resources_creation_times)

        if final_solver is not None:
            unassigned_agents = _safe_attr(critical_resources, "unassigned_agents_per_tol", None)
            unassigning_times = _safe_attr(critical_resources, "unassigning_times", None)
            if unassigning_times is not None:
                total_time_clusters_post += sum(unassigning_times)

            model_times_final = _safe_attr(final_solver, "model_times", None)
            resolution_times_final = _safe_attr(final_solver, "resolution_times", None)
            status_final = _safe_attr(final_solver, "status", None)

            if model_times_final is not None:
                total_time_clusters_post += sum(model_times_final)
            if resolution_times_final is not None:
                total_time_clusters_post += sum(resolution_times_final)

    final_delay = sum(a.delay for a in env.agents)

    out = {
        f"refinement levels{suffix}": _safe_attr(env, "refinement_levels", None),
        f"E threshold{suffix}": _safe_attr(env, "E_abs_threshold", None),
        f"R threshold{suffix}": _safe_attr(env, "R_max_threshold", None),

        f"n clusters{suffix}": n_clusters,
        f"similarity index{suffix}": _safe_attr(env, "similarity_index", None),
        f"cluster similarity indexes{suffix}": _dump(_safe_attr(env, "cluster_similarity_indexes", None)),
        f"cluster congestion indexes{suffix}": _dump(_safe_attr(env, "cluster_congestion_indexes", None)),
        f"cluster congestion indexes absolute{suffix}": _dump(_safe_attr(env, "cluster_congestion_indexes_absolute", None)),
        f"cluster congestion ratio max{suffix}": _dump(_safe_attr(env, "cluster_congestion_ratio_max", None)),
        f"similarity matrix time{suffix}": _safe_attr(env, "matrix_time", None),
        f"nj time{suffix}": _safe_attr(env, "nj_time", None),

        f"n agents per cluster{suffix}": _dump(n_agents_per_cluster),
        f"od pairs per cluster{suffix}": _dump(od_pairs_per_cluster),

        f"model times clusters{suffix}": _dump(model_times_clusters),
        f"resolution times clusters{suffix}": _dump(resolution_times_clusters),
        f"clusters status{suffix}": _dump(clusters_status),
        f"UBs clusters{suffix}": _dump(UBs_clusters),
        f"LBs clusters{suffix}": _dump(LBs_clusters),

        f"critical resources creation times{suffix}": _dump(critical_resources_creation_times),
        f"unassigned agents{suffix}": _dump(unassigned_agents),
        f"unassigning agents times{suffix}": _dump(unassigning_times),

        f"model times final{suffix}": _dump(model_times_final),
        f"resolution times final{suffix}": _dump(resolution_times_final),
        f"status final{suffix}": status_final,

        f"UB clusters{suffix}": UB_clusters,
        f"LB clusters{suffix}": LB_clusters,
        f"final delay{suffix}": int(final_delay),
        f"total time clusters + post{suffix}": float(total_time_clusters_post),
    }
    return out


# -----------------------------
# Main function
# -----------------------------
def save_results(
    env_1: Environment,
    cluster_solvers_1: list[Heuristic_Solver],
    critical_resources_1: Critical_Resources | None,
    final_solver_1: Heuristic_Solver | None,
    env_2: Environment,
    cluster_solvers_2: list[Heuristic_Solver],
    critical_resources_2: Critical_Resources | None,
    final_solver_2: Heuristic_Solver | None,
    complete_solver: Heuristic_Solver,
    seed: int,
    df: DataFrame,
):
    # -------------------------
    # Parameters / instance info
    # -------------------------
    grid_side = env_1.grid_side
    n_quadrants = env_1.n_quadrants
    n_pairs_per_quadrant = env_1.n_pairs_per_quadrant
    n_agents = len(env_1.agents)
    max_cluster_size = env_1.max_cluster_size
    offset = env_1.offset
    k = env_1.k
    restrict_paths_to_quadrant = env_1.restrict_paths_to_quadrant
    env_time = env_1.set_time

    # -------------------------
    # Complete solution block
    # -------------------------
    model_times_complete = _safe_attr(complete_solver, "model_times", None)
    resolution_times_complete = _safe_attr(complete_solver, "resolution_times", None)
    status_complete = _safe_attr(complete_solver, "status", None)

    UB_complete = _safe_objval(complete_solver)
    LB_complete = _safe_objbound(complete_solver)
    gap_complete = _safe_attr(complete_solver, "gap", None)

    incumbent = _safe_attr(complete_solver, "incumbent", None)
    incumbent_times = _safe_attr(incumbent, "times", []) if incumbent is not None else []
    incumbent_solutions = _safe_attr(incumbent, "solutions", []) if incumbent is not None else []

    total_time_complete = 0.0
    if model_times_complete is not None:
        total_time_complete += sum(model_times_complete)
    if resolution_times_complete is not None:
        total_time_complete += sum(resolution_times_complete)

    # -------------------------
    # Cluster blocks (1 and 2)
    # -------------------------
    block_1 = _extract_cluster_block(env_1, cluster_solvers_1, critical_resources_1, final_solver_1, suffix=" 1")
    block_2 = _extract_cluster_block(env_2, cluster_solvers_2, critical_resources_2, final_solver_2, suffix=" 2")

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
        "restrict paths to quadrant": bool(restrict_paths_to_quadrant),
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

    # Merge blocks
    row.update(block_1)
    row.update(block_2)

    # Append to df (ensures missing columns become NaN if any)
    df.loc[len(df)] = row



def read_results():
    pass