from datetime import datetime
from pandas import DataFrame

from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from mcpa.elements.mcpa_environment import MCPA_Environment


def run_mcpa_scalability(
    base_grid_side: int,
    n_pairs_per_quadrant: int,
    max_cluster_size: int,
    resources_capacity: int,
    offset: int,
    df: DataFrame,
    starting_seed: int,
    q_range: list[int],
    n_iterations: int,
    sim_method: tuple[str, float| None]
):
    seed = starting_seed

    for n_quadrants in q_range:
        grid_side = 2 * base_grid_side if n_quadrants <= 4 else 3 * base_grid_side
        for i in range(n_iterations):
            env = MCPA_Environment(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, resources_capacity, offset, 10, seed=seed, sim_method=sim_method)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   cap={env.resources_capacity}   seed={seed}   method={sim_method[0]}   iteration={i}"), end="   ")

            global_solver = MCPA_Heuristic_Solver(env.G, env.elements)
            global_solver.solve()

            env.solve_clusters()

            critical_resources = None
            repair_solver = None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = MCPA_Critical_Resources(env.G, env.elements)
                if not critical_resources.is_initially_feasible:
                    repair_solver = MCPA_Heuristic_Solver(env.G, env.elements, critical_resources)
                    repair_solver.solve()

            save_results(env, global_solver, seed, df, critical_resources, repair_solver)
            seed += 1
            df.to_csv(f"results/mcpa/small_instances_{sim_method[0]}.csv", index=False)


def save_results(
    env: MCPA_Environment,
    global_solver: MCPA_Heuristic_Solver,
    seed: int,
    df: DataFrame,
    critical_resources: MCPA_Critical_Resources | None = None,
    repair_solver: MCPA_Heuristic_Solver | None = None
):
    # INSTANCE PARAMETER
    grid_side = env.grid_side
    n_quadrants = env.n_quadrants
    n_pairs_per_quadrant = env.n_elements_per_quadrant
    n_agents = len(env.agents)
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

    # CLUSTERS FEATURES (BEFORE OPTMIZATION)
    n_clusters = len(env.clusters)
    n_agents_per_cluster = [c.n_agents for c in env.clusters]
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

    # HEURISTIC RESOLUTION
    model_times_clusters = [sum(cl.model_times) for cl in env.clusters_solvers]
    resolution_time_clusters = [sum(cl.resolution_times) for cl in env.clusters_solvers]
    LB_heuristic = None
    UB_heuristic = None
    gap = None
    unassigned_agents = None
    final_unassigned_agents = None
    model_times_repair = None
    resolution_times_repair = None
    final_tolerance = None

    time_heuristic = env.matrix_time + env.nj_time + sum(sum(hs.model_times) for hs in env.clusters_solvers) + sum(sum(hs.resolution_times) for hs in env.clusters_solvers)
    if critical_resources is not None:
        time_heuristic += sum(critical_resources.creation_times)
        if repair_solver is not None:
            unassigned_agents = critical_resources.unassigned_items_per_tol if not critical_resources.is_initially_feasible else [0]
            final_unassigned_agents = unassigned_agents[-1]
            model_times_repair = repair_solver.model_times
            resolution_times_repair = repair_solver.resolution_times
            final_tolerance = critical_resources.current_tol
            time_heuristic += sum(critical_resources.unassigning_times) + sum(repair_solver.model_times) + sum(
                repair_solver.resolution_times)

            LB_heuristic, UB_heuristic = compute_heuristic_bounds(env, critical_resources, repair_solver)

            # --- INIZIO NUOVO CONTROLLO ---
            if LB_heuristic is None or UB_heuristic is None:
                print(f"Impossibile trovare ObjVal/ObjBound. Nessun salvataggio per seed={seed}.")
                return  # Interrompe save_results; la riga non verrà aggiunta al DataFrame
            # --- FINE NUOVO CONTROLLO ---

            gap = 100 * (UB_heuristic - UB_global) / UB_global
            print(f"gap={gap}   speedup={round(time_global / time_heuristic, 2)}")
        else:
            delay = sum(a.delay for a in env.agents)
            LB_heuristic = delay
            UB_heuristic = delay
            gap = 100 * (UB_heuristic - UB_global) / UB_global
            print(f"gap={gap}   speedup={round(time_global / time_heuristic, 2)}")

    row = {
        "grid side": grid_side,
        "n quadrants": n_quadrants,
        "n pairs per quadrant": n_pairs_per_quadrant,
        "n agents": n_agents,
        "mean resource capacity": env.resources_capacity,
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

        "n clusters": n_clusters,
        "n agents per cluster": n_agents_per_cluster,
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
        "model times clusters": model_times_clusters,
        "resolution times clusters": resolution_time_clusters,
        "unassigned agents": unassigned_agents,
        "final unassigned agents": final_unassigned_agents,
        "model times repair": model_times_repair,
        "resolution times repair": resolution_times_repair,
        "final tolerance": final_tolerance,
        "LB heuristic": LB_heuristic,
        "UB heuristic": UB_heuristic,
        "time heuristic": time_heuristic,
        "gap": gap,
        "speedup": time_global / time_heuristic
    }
    df.loc[len(df)] = row




def compute_heuristic_bounds(
    env: MCPA_Environment,
    critical_resources: MCPA_Critical_Resources,
    repair_solver: MCPA_Heuristic_Solver
):
    try:
        repair_lb = repair_solver.m.ObjBound
        repair_ub = repair_solver.m.ObjVal
    except Exception:
        return None, None

    removed_agents = set(critical_resources.removed_items)
    fixed_delay = sum(a.delay for a in env.agents if a not in removed_agents)

    LB = fixed_delay + repair_lb
    UB = fixed_delay + repair_ub
    return LB, UB



