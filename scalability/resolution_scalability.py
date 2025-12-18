from pandas import DataFrame

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment
from utils.results_io import save_results


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
            print(f"\n\n=========================================================================================================\n\n")
            print(f"n quadrant = {n_quadrants}    n pairs per quadrant = {n_pairs_per_quadrant}    iteration {i}")

            env = Environment(grid_side,  n_pairs_per_quadrant * 5, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed)
            complete_solver = Heuristic_Solver(env.G, env.od_pairs, verbose=False)
            complete_solver.solve()
            print(f"\n\n=========================================================================================================\n\n")


            env.compute_clusters()
            print(f"{len(env.clusters)} clusters   ", end="")
            cluster_solvers = []
            all_clusters_ok = True
            for cluster in env.clusters:
                print(f"\n\n=========================================================================================================\n\nCLUSTER {cluster.id}\n\n")

                hs = Heuristic_Solver(env.G, cluster.od_pairs, verbose=False)
                hs.solve()
                print(f"\n\n=========================================================================================================\n\n")

                if hs.status != "OPTIMAL":
                    all_clusters_ok = False

                cluster_solvers.append(hs)

            if not all_clusters_ok:
                print()
                save_results(env, complete_solver, cluster_solvers, df, seed, None, None)
            else:
                critical_resources = Critical_Resources(env.G, env.od_pairs)
                if critical_resources.is_initially_feasible:
                    save_results(env, complete_solver, cluster_solvers, df, seed, critical_resources, None)
                else:
                    critical_resources.unassign_agents()
                    final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
                    final_solver.solve()
                    save_results(env, complete_solver, cluster_solvers, df, seed, critical_resources, final_solver)

            seed += 1
        print()

    df.to_csv(f"results/prova.csv", index=False)

