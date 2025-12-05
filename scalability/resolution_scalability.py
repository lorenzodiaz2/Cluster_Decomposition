import pandas as pd

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment
from utils.results_io import save_results

def run_scalability(n_pairs_per_quadrant):
    df = get_data_frame()

    quadrant_range = range(2, 10)
    for n_quadrants in quadrant_range:
        print()
        grid_side = 41 if n_quadrants <= 4 else 62
        for i in range(5):
            print(f"\n\n=========================================================================================================\n\n")
            print(f"n quadrant = {n_quadrants}    iteration {i} ->  Setting env... ", end="")
            seed = i + 40

            env = Environment(grid_side,  n_pairs_per_quadrant * 5, n_quadrants, n_pairs_per_quadrant, -1, 10, seed=seed, restrict_paths_to_quadrant=True)
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

    df.to_csv(f"{n_pairs_per_quadrant}_test_2-9.csv", index=False)



def get_data_frame():
    return pd.DataFrame({
        "grid side": pd.Series(dtype="int"),
        "n quadrants": pd.Series(dtype="int"),
        "n pairs": pd.Series(dtype="int"),
        "n agents": pd.Series(dtype="int"),
        "max cluster size": pd.Series(dtype="int"),
        "offset": pd.Series(dtype="int"),
        "k": pd.Series(dtype="int"),
        "seed": pd.Series(dtype="int"),
        "env time": pd.Series(dtype="float"),
        "model times complete": pd.Series(dtype="object"),
        "resolution times complete": pd.Series(dtype="object"),
        "status complete": pd.Series(dtype="object"),
        "n clusters": pd.Series(dtype="int"),
        "similarity index": pd.Series(dtype="float"),
        "cluster similarity indexes": pd.Series(dtype="object"),
        "similarity matrix time": pd.Series(dtype="float"),
        "nj time": pd.Series(dtype="float"),
        "n agents per cluster": pd.Series(dtype="object"),
        "od pairs per cluster": pd.Series(dtype="object"),
        "model times clusters": pd.Series(dtype="object"),
        "resolution times clusters": pd.Series(dtype="object"),
        "clusters status": pd.Series(dtype="object"),
        "UBs clusters": pd.Series(dtype="object"),
        "LBs clusters": pd.Series(dtype="object"),
        "critical resources creation times": pd.Series(dtype="object"),
        "unassigned agents": pd.Series(dtype="object"),
        "unassigning agents times": pd.Series(dtype="object"),
        "model times final": pd.Series(dtype="object"),
        "resolution times final": pd.Series(dtype="object"),
        "status final": pd.Series(dtype="object"),
        "UB complete": pd.Series(dtype="float"),
        "LB complete": pd.Series(dtype="float"),
        "UB clusters": pd.Series(dtype="float"),
        "LB clusters": pd.Series(dtype="float"),
        "final delay": pd.Series(dtype="int"),
        "total time complete": pd.Series(dtype="float"),
        "total time clusters + post": pd.Series(dtype="float")
    })




