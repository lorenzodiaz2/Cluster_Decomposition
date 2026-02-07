from datetime import datetime
from pandas import DataFrame

from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from mcpa.elements.mcpa_environment import MCPA_Environment
from utils.mcpa_results_io import save_mcpa_results


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

            env = MCPA_Environment(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed)

            complete_solver = MCPA_Heuristic_Solver(env.G, env.elements)
            complete_solver.solve()

            env.solve_clusters()

            critical_resources = None
            final_solver = None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = MCPA_Critical_Resources(env.G, env.elements)
                if not critical_resources.is_initially_feasible:
                    final_solver = MCPA_Heuristic_Solver(env.G, env.elements, critical_resources)
                    final_solver.solve()

            save_mcpa_results(env, critical_resources, final_solver, complete_solver, seed, df)
            seed += 1
            df.to_csv(f"results/mcpa/mcpa_results.csv", index=False)
        print()

