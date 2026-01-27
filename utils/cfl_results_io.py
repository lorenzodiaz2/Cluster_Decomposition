from typing import Any

from pandas import DataFrame

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.cfl_critical_resources import CFL_Critical_Resources
from cfl.solver.cfl_heuristic_solver import CFL_Heuristic_Solver, total_solution_cost
from utils.mcpa_results_io import _safe_attr, _safe_objVal, _safe_objBound, _dump



def _extract_cluster_block(
    env: CFL_Environment,
    UB_complete: float,
    critical_resources: CFL_Critical_Resources | None,
    final_solver: CFL_Heuristic_Solver | None,
    is_ss: bool | None = True
) -> dict[str, Any]:

    # cluster stats
    n_clusters = len(env.clusters)
    n_client_per_clusters = [len(c.elements) for c in env.clusters]
    n_facilities_per_cluster = [len(c.facilities) for c in env.clusters]

    clients_per_clusters = [[client.id for client in c.elements] for c in env.clusters]
    facilities_per_clusters = [[f.id for f in c.facilities] for c in env.clusters]

    model_times_clusters = []
    resolution_times_clusters = []
    clusters_status = []


    for hs in env.clusters_solvers:
        model_times_clusters.append(_safe_attr(hs, "model_times", [None])[0] if _safe_attr(hs, "model_times", None) else None)
        resolution_times_clusters.append(_safe_attr(hs, "resolution_times", [None])[0] if _safe_attr(hs, "resolution_times", None) else None)
        clusters_status.append(_safe_attr(hs, "status", None))


    critical_resources_creation_times = None

    unassigned_items = None
    unassigning_items_times = None

    model_times_final = None
    resolution_times_final = None
    status_final = None

    total_time_heuristic = (env.matrix_time or 0.0) + (env.nj_time or 0.0)

    for hs in env.clusters_solvers:
        total_time_heuristic += sum(_safe_attr(hs, "model_times", []) or [])
        total_time_heuristic += sum(_safe_attr(hs, "resolution_times", []) or [])

    cluster_has_tl = any(s == "TIME_LIMIT" for s in (clusters_status or []))
    total_cost = total_solution_cost(env.G, env.elements, env.facilities)
    gap = None

    if cluster_has_tl:
        UB_final = None
        LB_final = None
    else:
        if critical_resources is not None:
            critical_resources_creation_times = _safe_attr(critical_resources, "creation_times", None)
            total_time_heuristic += sum(critical_resources_creation_times or [])

            if final_solver is not None:
                unassigned_items = _safe_attr(critical_resources, "unassigned_items_per_tol", None) if is_ss else sum(critical_resources.removed_units_by_client.values())
                unassigning_items_times = _safe_attr(critical_resources, "unassigning_times", None)
                total_time_heuristic += sum(unassigning_items_times or [])

                model_times_final = _safe_attr(final_solver, "model_times", None)
                resolution_times_final = _safe_attr(final_solver, "resolution_times", None)
                status_final = _safe_attr(final_solver, "status", None)

                fixed_cost = final_solver.fixed_cost_before
                ub = _safe_objVal(final_solver)
                lb = _safe_objBound(final_solver)

                UB_final = fixed_cost + ub if ub is not None else None
                LB_final = fixed_cost + lb if lb is not None else None

                total_time_heuristic += sum(model_times_final or [])
                total_time_heuristic += sum(resolution_times_final or [])

                gap = 100 * (UB_final - UB_complete) / UB_complete
            else:
                gap = 0.
                UB_final = total_cost
                LB_final = total_cost
        else:
            UB_final = total_cost
            LB_final = total_cost


    out = {
        f"n clusters": n_clusters,
        f"similarity index": _safe_attr(env, "similarity_index", None),
        f"cluster similarity indexes": _dump(_safe_attr(env, "cluster_similarity_indexes", None)),
        f"cluster congestion indexes": _dump(_safe_attr(env, "cluster_congestion_indexes", None)),
        f"cluster congestion indexes absolute": _dump(_safe_attr(env, "cluster_congestion_indexes_absolute", None)),

        f"global congestion absolute": _dump(_safe_attr(env, "global_congestion_index_absolute", None)),
        f"global congestion ratio max": _dump(_safe_attr(env, "global_congestion_ratio_max", None)),
        f"cross congestion absolute": _dump(_safe_attr(env, "cross_congestion_index_absolute", None)),
        f"cross congestion rate": _dump(_safe_attr(env, "cross_congestion_rate", None)),
        f"cross congestion share": _dump(_safe_attr(env, "cross_congestion_share", None)),

        f"cluster congestion ratio max": _dump(_safe_attr(env, "cluster_congestion_ratio_max", None)),
        f"similarity matrix time": _safe_attr(env, "matrix_time", None),
        f"nj time": _safe_attr(env, "nj_time", None),
        f"n clients per clusters": _dump(n_client_per_clusters),
        f"n facilities per clusters": _dump(n_facilities_per_cluster),
        f"clients per clusters": _dump(clients_per_clusters),
        f"facilities per clusters": _dump(facilities_per_clusters),

        f"model times clusters": _dump(model_times_clusters),
        f"resolution times clusters": _dump(resolution_times_clusters),
        f"clusters status": _dump(clusters_status),

        f"critical resources creation times": _dump(critical_resources_creation_times),

        f"{"unassigned clients" if is_ss else "unassigned demands"}": _dump(unassigned_items),
        f"{"unassigning clients times" if is_ss else "unassigning demands times"}": _dump(unassigning_items_times),

        f"model times final": _dump(model_times_final),
        f"resolution times final": _dump(resolution_times_final),
        f"status final": status_final,
        f"UB final": UB_final,
        f"LB final": LB_final,
        f"gap": gap,
        f"total time heuristic": float(total_time_heuristic),
    }
    print(f"t = {round(total_time_heuristic, 2)}  LB = {LB_final}   UB = {UB_final}    ({n_clusters})")
    return out




def save_cfl_results(
    env: CFL_Environment,
    critical_resources: CFL_Critical_Resources | None,
    final_solver: CFL_Heuristic_Solver | None,
    complete_solver: CFL_Heuristic_Solver,
    seed: int,
    df: DataFrame,
    is_ss: bool
):
    # -------------------------
    # Parameters / instance info
    # -------------------------
    grid_side = env.grid_side
    n_quadrants = env.n_quadrants
    n_clients_per_quadrant = env.n_elements_per_quadrant
    n_facilities_per_quadrant = env.n_facilities_per_quadrant
    single_client_demand = env.elements[0].demand
    single_facility_activation_cost = env.facilities[0].activation_cost
    single_facility_capacity = env.facilities[0].capacity
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
    # Cluster block
    # -------------------------
    block = _extract_cluster_block(env, UB_complete, critical_resources, final_solver, is_ss)

    # -------------------------
    # Build row dict following your DF schema
    # -------------------------
    row = {
        # PARAMETRI DI MODELLO
        "grid side": int(grid_side),
        "n quadrants": int(n_quadrants),
        "n clients per quadrant": int(n_clients_per_quadrant),
        "n facilities per quadrant": int(n_facilities_per_quadrant),
        "single client demand": int(single_client_demand),
        "single facility activation cost": int(single_facility_activation_cost),
        "single facility capacity": int(single_facility_capacity),
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



