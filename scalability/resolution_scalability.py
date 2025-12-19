from pandas import DataFrame

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment
from utils.results_io import save_results

def solve_clusters(env: Environment, refinement_levels: int = 0):
    env.compute_clusters(refinement_levels=refinement_levels)
    cluster_solvers = []
    all_clusters_ok = True
    for cluster in env.clusters:
        hs = Heuristic_Solver(env.G, cluster.od_pairs, verbose=False)
        hs.solve()
        if hs.status != "OPTIMAL":
            all_clusters_ok = False
        cluster_solvers.append(hs)

    return cluster_solvers, all_clusters_ok


def solve_final(env, all_clusters_ok):
    critical_resources = None
    final_solver = None
    if all_clusters_ok:
        critical_resources = Critical_Resources(env.G, env.od_pairs)
        if not critical_resources.is_initially_feasible:
            critical_resources.unassign_agents()
            final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
            final_solver.solve()
    return critical_resources, final_solver


def run_scalability(
    base_grid_side: int,
    n_pairs_per_quadrant: int,
    offset: int,
    df: DataFrame,
    starting_seed: int
):
    seed = starting_seed

    quadrant_range = range(2, 10)
    for n_quadrants in quadrant_range:
        grid_side = 2 * base_grid_side + 1 if n_quadrants <= 4 else 3 * base_grid_side + 2
        for i in range(10):
            print(f"n quadrant = {n_quadrants}    n pairs per quadrant = {n_pairs_per_quadrant}    iteration {i}   ", end="")

            env_1 = Environment(grid_side,  n_pairs_per_quadrant * 5, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed)

            complete_solver = Heuristic_Solver(env_1.G, env_1.od_pairs, verbose=False)
            complete_solver.solve()
            augmented_T_times = len(complete_solver.model_times) - 1

            cluster_solvers_1, all_clusters_ok_1 = solve_clusters(env_1)
            critical_resources_1, final_solver_1 = solve_final(env_1, all_clusters_ok_1)

            # =======================================================================================================================

            env_2 = Environment(grid_side, n_pairs_per_quadrant * 5, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed)
            for _ in range(augmented_T_times):
                for od in env_2.od_pairs:
                    od.delay_shortest_paths(od.T + 1)
                    od.T += 1

            cluster_solvers_2, all_clusters_ok_2 = solve_clusters(env_2, 1)
            critical_resources_2, final_solver_2 = solve_final(env_2, all_clusters_ok_2)


            save_results(env_1, cluster_solvers_1, critical_resources_1, final_solver_1, env_2, cluster_solvers_2, critical_resources_2, final_solver_2, complete_solver, seed, df)
            print()
            seed += 1
        print()

    df.to_csv(f"results/prova.csv", index=False)

