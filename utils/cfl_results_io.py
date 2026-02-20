import math
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.cfl_critical_resources import CFL_Critical_Resources
from cfl.solver.cfl_heuristic_solver import CFL_Heuristic_Solver, total_solution_cost
from utils.mcpa_results_io import _safe_attr, _safe_objVal, _safe_objBound, _dump
from utils.read_instance import fmt_int, fmt_float, get_sum_from_array_string, get_last_from_array_string


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
    total_cost = total_solution_cost(env.elements, env.facilities)
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
                gap = 0. if UB_complete == total_cost else 100 * (total_cost - UB_complete) / UB_complete
                UB_final = total_cost
                LB_final = total_cost
        else:
            UB_final = total_cost
            LB_final = total_cost

    unassigned_string = "unassigned clients" if is_ss else "unassigned demands"
    unassigned_times_string = "unassigning clients times" if is_ss else "unassigning demands times"
    n_open_facilities_final = sum(1 for f in env.facilities if f.is_open)

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

        f"{unassigned_string}": _dump(unassigned_items),
        f"{unassigned_times_string}": _dump(unassigning_items_times),

        f"model times final": _dump(model_times_final),
        f"resolution times final": _dump(resolution_times_final),
        f"status final": status_final,
        f"UB final": UB_final,
        f"LB final": LB_final,
        f"n open facilities final": n_open_facilities_final,
        f"% n open facilities final": round(100 * n_open_facilities_final / (env.n_facilities_per_quadrant * env.n_quadrants), 2),
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

    n_open_facilities_complete = sum(1 for var in complete_solver.x.values() if var.X > 0.5)

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
        "n open facilities complete": n_open_facilities_complete,
        "gap complete": float(gap_complete) if gap_complete is not None else None,
        "incumbent times": _dump(incumbent_times),
        "incumbent solutions": _dump(incumbent_solutions),
        "total time complete": float(total_time_complete),
    }

    row.update(block)

    df.loc[len(df)] = row


def save_cfl_tex_tables():
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

    df = pd.read_csv("results/cfl/ss/sscfl_results.csv")
    df['unassigning clients times'] = df['unassigning clients times'].astype(str)
    df['model times final'] = df['model times final'].astype(str)
    df['resolution times final'] = df['resolution times final'].astype(str)
    df['critical resources creation times'] = df['critical resources creation times'].astype(str)
    df['unassigned clients'] = df['unassigned clients'].astype(str)

    for offset in [10]:
        df_out = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        for n_clients_per_quadrant in [125, 150, 175, 200, 225]:
            for n_quadrants in [2, 3, 4, 6, 7, 9]:
                n_instances = 0
                complete_times = []
                heuristic_times = []
                gaps = []
                n_time_limit_complete = 0
                n_time_limit_heuristic = 0
                speeds_up = []
                repair_times = []
                unassigned_clients = []
                n_clusters = []
                cluster_creation_times = []

                for _, row in df.iterrows():
                    _grid_side = int(row["grid side"])
                    _n_quadrants = int(row["n quadrants"])
                    _n_clients_per_quadrant = int(row["n clients per quadrant"])
                    _max_cluster_size = int(row["max cluster size"])
                    _offset = int(row["offset"])
                    _k = int(row["k"])
                    _seed = int(row["seed"])
                    _status_complete = row["status complete"]
                    _clusters_status = row["clusters status"]
                    _complete_time = float(row["total time complete"])
                    _heuristic_time = float(row["total time heuristic"])
                    _status_final = row["status final"]
                    _n_clusters = int(row["n clusters"])
                    _cluster_creation_time = 100 * (float(row["similarity matrix time"]) + float(row["nj time"])) / _heuristic_time

                    _repair_time = 0

                    if row["critical resources creation times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["critical resources creation times"])

                    if row["unassigning clients times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["unassigning clients times"])

                    if row["model times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["model times final"])

                    if row["resolution times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["resolution times final"])

                    _unassigned_clients = 0
                    if row["unassigned clients"] != "nan":
                        _unassigned_clients += get_last_from_array_string(row["unassigned clients"])

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
                    unassigned_clients.append(100 * _unassigned_clients / (_n_clients_per_quadrant * _n_quadrants))
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
                median_unassigned_clients = round(np.median(unassigned_clients), 2) if len(unassigned_clients) > 0 else 0
                median_n_clusters = np.median(n_clusters) if len(n_clusters) > 0 else 0
                median_clusters_creation_time = np.median(cluster_creation_times) if len(cluster_creation_times) > 0 else 0

                df_out.loc[len(df_out)] = [n_clients_per_quadrant, n_quadrants, median_complete_times, n_time_limit_complete, median_heuristic_times, n_time_limit_heuristic, median_clusters_creation_time, median_repair_times, median_unassigned_clients, median_n_clusters, median_gaps, median_speed_up]

        df_out.to_latex(
            f"results/cfl/ss/tex/summary_{offset}.tex",
            index=False,
            formatters=column_formatters
        )


    df = pd.read_csv("results/cfl/ms/mscfl_results.csv")
    df['unassigning demands times'] = df['unassigning demands times'].astype(str)
    df['model times final'] = df['model times final'].astype(str)
    df['resolution times final'] = df['resolution times final'].astype(str)
    df['critical resources creation times'] = df['critical resources creation times'].astype(str)
    df['unassigned demands'] = df['unassigned demands'].astype(str)


    for offset in [10]:
        df_out = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        for n_clients_per_quadrant in [125, 150, 175, 200, 225]:
            for n_quadrants in [2, 3, 4, 6, 7, 9]:
                n_instances = 0
                complete_times = []
                heuristic_times = []
                gaps = []
                n_time_limit_complete = 0
                n_time_limit_heuristic = 0
                speeds_up = []
                repair_times = []
                unassigned_demands = []
                n_clusters = []
                cluster_creation_times = []

                for _, row in df.iterrows():
                    _grid_side = int(row["grid side"])
                    _n_quadrants = int(row["n quadrants"])
                    _n_clients_per_quadrant = int(row["n clients per quadrant"])
                    _max_cluster_size = int(row["max cluster size"])
                    _offset = int(row["offset"])
                    _k = int(row["k"])
                    _seed = int(row["seed"])
                    _status_complete = row["status complete"]
                    _clusters_status = row["clusters status"]
                    _complete_time = float(row["total time complete"])
                    _heuristic_time = float(row["total time heuristic"])
                    _status_final = row["status final"]
                    _n_clusters = int(row["n clusters"])
                    _cluster_creation_time = 100 * (float(row["similarity matrix time"]) + float(row["nj time"])) / _heuristic_time

                    _repair_time = 0

                    if row["critical resources creation times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["critical resources creation times"])

                    if row["unassigning demands times"] != "nan":
                        _repair_time += get_sum_from_array_string(row["unassigning demands times"])

                    if row["model times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["model times final"])

                    if row["resolution times final"] != "nan":
                        _repair_time += get_sum_from_array_string(row["resolution times final"])

                    _unassigned_demand = 0
                    if row["unassigned demands"] != "nan":
                        _unassigned_demand += get_last_from_array_string(row["unassigned demands"])

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
                    unassigned_demands.append(100 * _unassigned_demand / (_n_clients_per_quadrant * _n_quadrants * 20))
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
                median_unassigned_clients = round(np.median(unassigned_demands), 2) if len(unassigned_demands) > 0 else 0
                median_n_clusters = np.median(n_clusters) if len(n_clusters) > 0 else 0
                median_clusters_creation_time = np.median(cluster_creation_times) if len(cluster_creation_times) > 0 else 0

                df_out.loc[len(df_out)] = [n_clients_per_quadrant, n_quadrants, median_complete_times, n_time_limit_complete, median_heuristic_times, n_time_limit_heuristic, median_clusters_creation_time, median_repair_times, median_unassigned_clients, median_n_clusters, median_gaps, median_speed_up]

        df_out.to_latex(
            f"results/cfl/ms/tex/summary_{offset}.tex",
            index=False,
            formatters=column_formatters
        )


