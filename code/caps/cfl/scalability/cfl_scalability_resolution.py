from datetime import datetime
from typing import Callable

import numpy as np
from pandas import DataFrame

from caps.cfl.elements.cfl_environment import CFL_Environment
from caps.cfl.solver.cfl_critical_resources import CFL_Critical_Resources
from caps.cfl.solver.cfl_heuristic_solver import CFL_Heuristic_Solver, total_solution_cost
from caps.cfl.solver.multi_source.mscfl_critical_resources import MSCFL_Critical_Resources
from caps.cfl.solver.multi_source.mscfl_heuristic_solver import MSCFL_Heuristic_solver
from caps.cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from caps.cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_Solver


def run_cfl (
    base_grid_side: int,
    n_facilities: int,
    max_cluster_size: int,
    offset: int,
    k,
    df: DataFrame,
    starting_seed: int,
    q_range: list[int],
    n_iterations: int,
    env_maker: Callable[..., CFL_Environment],
    solver_maker: Callable[..., CFL_Heuristic_Solver],
    cluster_solver_maker: Callable[..., CFL_Heuristic_Solver],
    critical_resource_maker: Callable[..., CFL_Critical_Resources],
    output_file: str
):
    seed = starting_seed

    for n_quadrants in q_range:
        grid_side = 2 * base_grid_side if n_quadrants <= 4 else 3 * base_grid_side
        for i in range(n_iterations):
            print(datetime.now().strftime("%d-%m-%Y   %H:%M:%S    "), end="")

            env = env_maker(grid_side, max_cluster_size, n_quadrants, n_facilities, offset, k, seed)

            print(f"n quadrants={n_quadrants}    n clients per q={n_facilities * 5}    n fac per q={n_facilities}    offset={offset}   cluster size={max_cluster_size}   cap={env.resources_capacity}   seed={seed}    iteration {i}   ", end="")

            global_solver = solver_maker(env, None)
            global_solver.solve()

            env.solve_clusters(cluster_solver_maker)

            critical_resources = None
            repair_solver = None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = critical_resource_maker(env)
                if not critical_resources.is_initially_feasible:
                    repair_solver = solver_maker(env, critical_resources)
                    repair_solver.solve()

            save_results(env, global_solver, seed, df, critical_resources, repair_solver)
            seed += 1
            df.to_csv(output_file, index=False)




def run_sscfl(grid_side, n, max_cluster_size, offset, cap, k, df, seed, q_range, n_iterations, output_file):
    env_maker = lambda _grid_side, _max_cluster_size, _n_quadrants, _n, _offset, _k, _seed: CFL_Environment(
        grid_side=_grid_side,
        max_cluster_size=_max_cluster_size,
        n_quadrants=_n_quadrants,
        n_clients_per_quadrant=_n * 5,
        n_facilities_per_quadrant=_n,
        resources_capacities=cap,
        offset=_offset,
        k=_k,
        seed=_seed,
        sim_method=("exponential", 0.95),
        resource_weights="linear"
    )
    global_solver_maker = lambda env, cr: SSCFL_Heuristic_Solver(env.G, env.elements, env.facilities, cr)
    cluster_solver_maker = lambda env, cluster: SSCFL_Heuristic_Solver(env.G, cluster.elements, cluster.facilities)
    critical_resources_maker = lambda env: SSCFL_Critical_Resources(env.elements, env.facilities)

    run_cfl(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations, env_maker, global_solver_maker, cluster_solver_maker, critical_resources_maker, output_file)


def run_mscfl(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations, output_file):
    env_maker = lambda _grid_side, _max_cluster_size, _n_quadrants, _n, _offset, _k, _seed: CFL_Environment(
        grid_side=_grid_side,
        max_cluster_size=_max_cluster_size,
        n_quadrants=_n_quadrants,
        n_clients_per_quadrant=_n * 5,
        n_facilities_per_quadrant=_n,
        offset=_offset,
        k=_k,
        seed=_seed
    )
    global_solver_maker = lambda env, cr: MSCFL_Heuristic_solver(env.G, env.elements, env.facilities, cr)
    cluster_solver_maker = lambda env, cluster: MSCFL_Heuristic_solver(env.G, cluster.elements, cluster.facilities)
    critical_resources_maker = lambda env: MSCFL_Critical_Resources(env.elements, env.facilities)

    run_cfl(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations, env_maker, global_solver_maker, cluster_solver_maker, critical_resources_maker, output_file)




def save_results(
    env: CFL_Environment,
    global_solver: CFL_Heuristic_Solver,
    seed: int,
    df: DataFrame,
    critical_resources: CFL_Critical_Resources | None = None,
    repair_solver: CFL_Heuristic_Solver | None = None
):
    # INSTANCE PARAMETER
    grid_side = env.grid_side
    n_quadrants = env.n_quadrants
    n_clients_per_quadrant = env.n_elements_per_quadrant
    n_facilities_per_quadrant = env.n_facilities_per_quadrant
    mean_client_demand = sum(c.demand for c in env.elements) / len(env.elements)
    mean_facility_activation_cost = sum(f.activation_cost for f in env.facilities) / len(env.facilities)
    mean_facility_capacity = sum(f.capacity for f in env.facilities) / len(env.facilities)
    max_cluster_size = env.max_cluster_size
    offset = env.offset
    k = env.k

    # GLOBAL SOLVER RESOLUTION
    time_global = sum(global_solver.model_times) + sum(global_solver.resolution_times)
    model_times_global = global_solver.model_times
    resolution_times_global = global_solver.resolution_times
    status_global = global_solver.status
    LB_global = global_solver.m.ObjBound
    UB_global = global_solver.m.ObjVal
    n_open_facilities_global = sum(1 for var in global_solver.x.values() if var.X > 0.5)

    # CLUSTERS FEATURES (BEFORE OPTMIZATION)
    n_clusters = len(env.clusters)
    n_clients_per_cluster = [len(c.elements) for c in env.clusters]
    n_facilities_per_cluster = [len(c.facilities) for c in env.clusters]
    similarity_index = env.similarity_index
    cluster_similarity_indexes = env.cluster_similarity_indexes
    min_cluster_similarity_index = min(env.cluster_similarity_indexes)
    max_mean_intercluster_similarity = env.max_mean_intercluster_similarity
    silhouette_score = env.silhouette_score
    cluster_congestion_indexes = env.cluster_congestion_indexes
    cluster_congestion_indexes_absolute = env.cluster_congestion_indexes_absolute
    cluster_congestion_ratio_max = env.cluster_congestion_ratio_max
    global_congestion_absolute = env.global_congestion_index_absolute
    global_congestion_ratio_max = env.global_congestion_ratio_max
    cross_congestion_absolute = env.cross_congestion_index_absolute
    cross_congestion_rate = env.cross_congestion_rate
    cross_congestion_share = env.cross_congestion_share
    normalized_macro_components = env.normalized_macro_components
    max_eigengap = env.max_eigengap
    clustering_distortion_ratio = env.clustering_distortion_ratio
    absorption_rate = max_cluster_size / (n_clients_per_quadrant * n_quadrants)
    min_number_clusters = (n_clients_per_quadrant * n_quadrants) / max_cluster_size
    expected_clusters = clustering_distortion_ratio * n_clusters
    forced_fragmentation = np.where(expected_clusters > 0, min_number_clusters / expected_clusters, 0)
    interaction_density = env.interaction_density
    similarity_gini = env.similarity_gini
    similarity_skewness = env.similarity_skewness
    mass_cv = env.mass_cv
    boundary_entities_ratio = env.boundary_entities_ratio
    partition_entropy = env.partition_entropy
    critical_resources_ratio = env.critical_resources_ratio
    congestion_gini = env.congestion_gini
    cross_cluster_footprint_density = env.cross_cluster_footprint_density
    average_footprint_breadth = env.average_footprint_breadth
    average_guaranteed_slack = env.average_guaranteed_slack
    space_to_mass_ratio = env.space_to_mass_ratio

    # HEURISTIC RESOLUTION
    model_times_clusters = [sum(cl.model_times) for cl in env.clusters_solvers]
    resolution_time_clusters = [sum(cl.resolution_times) for cl in env.clusters_solvers]
    LB_heuristic = None
    UB_heuristic = None
    gap = None
    unassigned_items = None
    final_unassigned_items = None
    model_times_repair = None
    resolution_times_repair = None
    final_tolerance = None
    total_cost = total_solution_cost(env.elements, env.facilities)


    time_heuristic = env.matrix_time + env.partition_time + sum(sum(hs.model_times) for hs in env.clusters_solvers) + sum(sum(hs.resolution_times) for hs in env.clusters_solvers)
    if critical_resources is not None:
        time_heuristic += sum(critical_resources.creation_times)
        if repair_solver is not None:
            unassigned_items = critical_resources.released_units_per_tol if not critical_resources.is_initially_feasible else [0]
            final_unassigned_items = unassigned_items[-1]
            model_times_repair = repair_solver.model_times
            resolution_times_repair = repair_solver.resolution_times
            final_tolerance = critical_resources.current_tol
            time_heuristic += sum(critical_resources.release_times) + sum(repair_solver.model_times) + sum(repair_solver.resolution_times)

            try:
                UB_heuristic = repair_solver.fixed_cost_before + repair_solver.m.ObjVal
                LB_heuristic = repair_solver.fixed_cost_before + repair_solver.m.ObjBound

            except AttributeError:
                print(f"Impossibile trovare ObjVal/ObjBound. Nessun salvataggio per seed={seed}.")
                return

            gap = 100 * (UB_heuristic - UB_global) / UB_global
            print(f"gap={gap}   speedup={round(time_global / time_heuristic, 2)}")
        else:
            LB_heuristic = total_cost
            UB_heuristic = total_cost
            gap = 100 * (UB_heuristic - UB_global) / UB_global
            print(f"gap={gap}   speedup={round(time_global / time_heuristic, 2)}")


    row = {
        "grid side": grid_side,
        "n quadrants": n_quadrants,
        "n clients per quadrant": n_clients_per_quadrant,
        "n facilities per quadrant": n_facilities_per_quadrant,
        "mean client demand": mean_client_demand,
        "mean facility capacity": mean_facility_capacity,
        "mean facility opening cost": mean_facility_activation_cost,
        "max cluster size": max_cluster_size,
        "offset": offset,
        "k": k,
        "seed": seed,

        "time global": time_global,
        "model times global": model_times_global,
        "resolution times global": resolution_times_global,
        "status global": status_global,
        "number of resolution": len(global_solver.resolution_times),
        "LB global": LB_global,
        "UB global": UB_global,
        "n open facilities global": n_open_facilities_global,
        "% n open facilities global": round(100 * n_open_facilities_global / len(env.facilities), 2),

        "n clusters": n_clusters,
        "n clients per cluster": n_clients_per_cluster,
        "n facilities per cluster": n_facilities_per_cluster,
        "similarity index": similarity_index,
        "cluster similarity indexes": cluster_similarity_indexes,
        "min cluster similarity index": min_cluster_similarity_index,
        "max mean intercluster similarity": max_mean_intercluster_similarity,
        "silhouette score": silhouette_score,
        "cluster congestion indexes": cluster_congestion_indexes,
        "cluster congestion indexes absolute": cluster_congestion_indexes_absolute,
        "cluster congestion ratio max": cluster_congestion_ratio_max,
        "global congestion absolute": global_congestion_absolute,
        "global congestion ratio max": global_congestion_ratio_max,
        "cross congestion absolute": cross_congestion_absolute,
        "cross congestion rate": cross_congestion_rate,
        "cross congestion share": cross_congestion_share,
        "normalized macro components": normalized_macro_components,
        "max eigengap": max_eigengap,
        "clustering distortion ratio": clustering_distortion_ratio,
        "absorption rate": absorption_rate,
        "forced fragmentation": forced_fragmentation,
        "interaction_density": interaction_density,
        "similarity_gini": similarity_gini,
        "similarity_skewness": similarity_skewness,
        "mass_cv": mass_cv,
        "boundary_entities_ratio": boundary_entities_ratio,
        "partition_entropy": partition_entropy,
        "critical_resources_ratio": critical_resources_ratio,
        "congestion_gini": congestion_gini,
        "cross_cluster_footprint_density": cross_cluster_footprint_density,
        "average_footprint_breadth": average_footprint_breadth,
        "average_guaranteed_slack": average_guaranteed_slack,
        "space_to_mass_ratio": space_to_mass_ratio,
        "spatial density": (n_clients_per_quadrant * n_quadrants) / (grid_side ** 2), # todo check
        "model times clusters": model_times_clusters,
        "resolution times clusters": resolution_time_clusters,
        "unassigned items": unassigned_items,
        "final unassigned items": final_unassigned_items,
        "model times repair": model_times_repair,
        "resolution times repair": resolution_times_repair,
        "final tolerance": final_tolerance,
        "LB heuristic": LB_heuristic,
        "UB heuristic": UB_heuristic,
        "n open facilities heuristic": sum(1 for f in env.facilities if f.is_open),
        "% n open facilities heuristic": round(100 * sum(1 for f in env.facilities if f.is_open) / len(env.facilities), 2),
        "time heuristic": time_heuristic,
        "gap": gap,
        "speedup": time_global / time_heuristic
    }
    df.loc[len(df)] = row



