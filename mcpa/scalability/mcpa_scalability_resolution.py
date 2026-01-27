from datetime import datetime

from pandas import DataFrame

from mcpa.elements.pair import OD_Pair
from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from mcpa.elements.mcpa_environment import MCPA_Environment
from utils.mcpa_results_io import save_mcpa_results

def solve_clusters(env: MCPA_Environment):
    env.compute_clusters(OD_Pair.compute_similarity)
    cluster_solvers = []
    all_clusters_ok = True
    for cluster in env.clusters:
        hs = MCPA_Heuristic_Solver(env.G, cluster.elements)
        hs.solve()
        if hs.status != "OPTIMAL":
            all_clusters_ok = False
        cluster_solvers.append(hs)

    return cluster_solvers, all_clusters_ok


def solve_final(env, all_clusters_ok):
    critical_resources = None
    final_solver = None
    if all_clusters_ok:
        critical_resources = MCPA_Critical_Resources(env.G, env.elements)
        if not critical_resources.is_initially_feasible:
            critical_resources.unassign_items()
            final_solver = MCPA_Heuristic_Solver(env.G, env.elements, critical_resources)
            final_solver.solve()
    return critical_resources, final_solver


def run_mcpa_scalability(
    base_grid_side: int,
    n_pairs_per_quadrant: int,
    max_cluster_size: int,
    offset: int,
    df: DataFrame,
    starting_seed: int,
    q_range: list[int],
    n_iterations: int
):
    seed = starting_seed

    for n_quadrants in q_range:
        grid_side = 2 * base_grid_side + 1 if n_quadrants <= 4 else 3 * base_grid_side + 2
        for i in range(n_iterations):
            print(datetime.now().strftime("%d-%m-%Y   %H:%M:%S    "), end="")

            print(f"n quadrants = {n_quadrants}    n pairs per quadrant = {n_pairs_per_quadrant}    offset = {offset}    iteration {i}   ", end="")

            env_1 = MCPA_Environment(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed)

            complete_solver = MCPA_Heuristic_Solver(env_1.G, env_1.elements)
            complete_solver.solve()

            cluster_solvers_1, all_clusters_ok_1 = solve_clusters(env_1)

            critical_resources_1, final_solver_1 = solve_final(env_1, all_clusters_ok_1)

            save_mcpa_results(env_1, critical_resources_1, final_solver_1, complete_solver, seed, df)
            seed += 1
        print()
        df.to_csv(f"results/test_20_750.csv", index=False)

