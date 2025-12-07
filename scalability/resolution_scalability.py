from pandas import DataFrame

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment
from utils.results_io import save_results

def run_scalability(
        base_grid_side: int,
        n_pairs_per_quadrant: int,
        restrict_paths_to_quadrant: bool,
        offset: int,
        df: DataFrame
):

    quadrant_range = range(2, 10)
    for n_quadrants in quadrant_range:
        print()
        grid_side = 2 * base_grid_side + 1 if n_quadrants <= 4 else 3 * base_grid_side + 2
        for i in range(10):
            print(f"\n\n=========================================================================================================\n\n")
            print(f"n quadrant = {n_quadrants}    n pairs per quadrant = {n_pairs_per_quadrant}    iteration {i} ->  Setting env... ", end="")
            seed = i + 40

            env = Environment(grid_side,  n_pairs_per_quadrant * 5, n_quadrants, n_pairs_per_quadrant, offset, 10, seed=seed, restrict_paths_to_quadrant=restrict_paths_to_quadrant)
            print("Done.\n\nCOMPLETA\n\n")
            complete_solver = Heuristic_Solver(env.G, env.od_pairs, output_flag=1)
            complete_solver.solve()
            print(f"\n\nSTATUS COMPLETE = {complete_solver.status}")
            print(f"\n\n=========================================================================================================\n\n")


            env.compute_clusters()
            print(f"{len(env.clusters)} clusters   ", end="")
            cluster_solvers = []
            all_clusters_ok = True
            for cluster in env.clusters:
                print(f"\n\n=========================================================================================================\n\nCLUSTER {cluster.id}\n\n")

                hs = Heuristic_Solver(env.G, cluster.od_pairs, output_flag=1)
                hs.solve()
                print(f"{hs.status} in {hs.model_times} + {hs.resolution_times}   ", end="")
                print(f"\n\n=========================================================================================================\n\n")

                if hs.status != "OPTIMAL":
                    all_clusters_ok = False
                else:
                    hs.assign_solutions()
                cluster_solvers.append(hs)
            # print("Done.   ", end="")

            if not all_clusters_ok:
                print()
                save_results(env, complete_solver, cluster_solvers, df, seed, None, None)
            else:
                critical_resources = Critical_Resources(env.G, env.od_pairs)
                if critical_resources.is_initially_feasible:
                    print("Solution is already feasible.")
                    save_results(env, complete_solver, cluster_solvers, df, seed, critical_resources, None)
                    continue
                else:
                    print("Unassigning agents... ", end="")
                    critical_resources.unassign_agents()
                    print("Done.   Solving final... ", end="")
                    final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
                    final_solver.solve()
                    final_solver.assign_solutions()
                    print("Done.")
                    save_results(env, complete_solver, cluster_solvers, df, seed, critical_resources, final_solver)

    df.to_csv(f"results/{base_grid_side}_test_2-9.csv", index=False)
