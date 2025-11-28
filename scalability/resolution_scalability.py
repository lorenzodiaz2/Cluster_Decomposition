import pandas as pd

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment
from utils.results_io import save_results

def run_scalability():
    df = pd.DataFrame(
        columns=["grid side", "n quadrants", "n pairs", "n agents", "max cluster size", "offset", "k", "env time",
                 "model times complete", "resolution times complete", "status complete", "UB complete", "LB complete",
                 "n clusters", "similarity matrix time", "nj time", "n agents per cluster", "model times clusters", "resolution times clusters",
                 "clusters status", "UBs clusters", "LBs clusters",
                 "critical resources creation times", "unassigned agents", "unassigning agents times",
                 "model times final", "resolution times final", "status final", "UB final", "LB final"]
    )

    quadrant_range = range(3, 6)
    for n_quadrants in quadrant_range:
        print()
        for i in range(5):
            print(f"n quadrant = {n_quadrants}    iteration {i} ->  Setting env... ", end="")
            env = Environment(30, 150, n_quadrants, 30, 0, 12, False)
            print("Done.   Solving complete... ", end="")
            complete_solver = Heuristic_Solver(env.G, env.od_pairs)
            complete_solver.solve()
            print("Done.   Solving clusters... ", end="")

            env.compute_clusters()
            cluster_solvers = []
            all_clusters_ok = True
            for cluster in env.clusters:
                hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs])
                hs.solve()
                if hs.status != "OPTIMAL":
                    all_clusters_ok = False
                else:
                    hs.assign_solutions()
                cluster_solvers.append(hs)
            print("Done.   ", end="")

            if not all_clusters_ok:
                print()
                save_results(env, complete_solver, cluster_solvers, i, df, None,None)
            else:
                critical_resources = Critical_Resources(env.G, env.od_pairs)
                if critical_resources.is_initially_feasible:
                    print("Solution is already feasible.")
                    save_results(env, complete_solver, cluster_solvers, i, df, critical_resources, None)
                    continue
                else:
                    print("Unassigning agents... ", end="")
                    critical_resources.unassign_agents()
                    print("Done.   Solving final... ", end="")
                    final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
                    final_solver.solve()
                    final_solver.assign_solutions()
                    print("Done.")
                    save_results(env, complete_solver, cluster_solvers, i, df, critical_resources, final_solver)

    df.to_csv("test.csv", index=False)





